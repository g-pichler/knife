import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import itertools
from model_utils import *



class VanillaSeq2seq(torch.nn.Module):
    def __init__(self, args):
        super(VanillaSeq2seq, self).__init__()
        self.args = args
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        if args.hidden_dim != args.dec_hidden_dim:
            self.linear = torch.nn.Linear(args.hidden_dim, args.dec_hidden_dim)
        self.decoder = DecoderRNN(args, args.dec_hidden_dim, args.number_of_tokens, args.number_of_layers)
        self.loss = torch.nn.NLLLoss(ignore_index=self.args.tokenizer.pad_token_id)

    def forward(self, input_tensor, labels, teacher_ratio):
        loss = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoded_words = [decoder_input]
        if self.args.hidden_dim != self.args.dec_hidden_dim:
            decoder_hidden = self.linear(encoder_hidden)
        else:
            decoder_hidden = encoder_hidden
        use_teacher_forcing = False if random.random() < teacher_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss += self.loss(decoder_output.squeeze(1), input_tensor[:, di])
                decoder_input = input_tensor[:, di].unsqueeze(-1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input

            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
                decoded_words.append(topi.squeeze(-1))
                loss += self.loss(decoder_output.squeeze(1), input_tensor[:, di])
        loss = loss / self.args.max_length

        if self.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()  # Update learning rate schedule
            self.encoder.zero_grad()
            self.decoder.zero_grad()
        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        return ({'generation_loss': loss}, decoded_words)

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
        return encoder_hidden

    def predict_style_vector(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
        return encoder_hidden

    def evaluate_for_style_transfert(self, input_tensor, style):
        loss = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoded_words = [decoder_input]
        decoder_hidden = encoder_hidden

        for di in range(self.args.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
            decoded_words.append(topi.squeeze(-1))
            loss += self.loss(decoder_output.squeeze(1), input_tensor[:, di])

        loss = loss / self.args.max_length

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        return ({'generation_loss': loss}, decoded_words)




class BaselineDisentanglement(torch.nn.Module):
    def __init__(self, args):
        super(BaselineDisentanglement, self).__init__()
        self.args = args
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.decoder = DecoderRNN(args, args.style_dim + args.content_dim, args.number_of_tokens, args.number_of_layers)
        self.loss = torch.nn.NLLLoss(ignore_index=self.args.tokenizer.pad_token_id)
        self.proj_style = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                        nn.Linear(args.hidden_dim, args.style_dim))
        self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                          nn.Linear(args.hidden_dim, args.content_dim))

        # Adversarial Classifiers
        self.adv_classifier_style = Classifier(args.content_dim, args.number_of_styles)
        self.mul_classifier_style = Classifier(args.style_dim, args.number_of_styles)

        # Loss : paper multipliers
        self.mul_style = 0
        self.adv_style = self.args.adv_style


    def forward(self, input_tensor, labels, teacher_ratio):
        # Update strategy is as follow :
        #   - update classifiers for style without encoder
        #   - update complete loss

        # Update Classifiers
        free_params(self.adv_classifier_style)
        frozen_params(self.mul_classifier_style)
        frozen_params(self.proj_style)
        frozen_params(self.proj_content)
        frozen_params(self.encoder)
        frozen_params(self.decoder)

        loss_classifier_adv_style = 0

        golden_tensor = input_tensor.clone()
        if self.args.add_noise:
            input_tensor = corrupt_input(self,input_tensor)

        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        style_content_pred = self.adv_classifier_style(content)
        loss_classifier_adv_style += self.adv_style * self.loss(style_content_pred, labels)

        loss = loss_classifier_adv_style
        if self.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mul_classifier_style.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.adv_classifier_style.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()  # Update learning rate schedule
            self.mul_classifier_style.zero_grad()
            self.adv_classifier_style.zero_grad()

        # Update complete loss
        frozen_params(self.adv_classifier_style)
        free_params(self.mul_classifier_style)
        free_params(self.proj_style)
        free_params(self.proj_content)
        free_params(self.encoder)
        free_params(self.decoder)

        loss_gen = 0
        loss_mul_style = 0
        loss_adv_style = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style is in style space
        style = self.proj_style(encoder_hidden)
        style_pred = self.mul_classifier_style(style)
        loss_mul_style += self.mul_style * self.loss(style_pred, labels) # set à 0

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        style_content_pred = self.adv_classifier_style(content)
        loss_adv_style += - self.adv_style * self.loss(style_content_pred, labels)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        # Sentence Generation
        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoder_hidden = encoder_hidden
        use_teacher_forcing = False if random.random() < teacher_ratio else False

        decoded_words = [decoder_input]

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss_gen += self.loss(decoder_output.squeeze(1), golden_tensor[:, di]) / self.args.max_length
                decoder_input = golden_tensor[:, di].unsqueeze(-1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
                decoded_words.append(topi.squeeze(-1))
                loss_gen += self.loss(decoder_output.squeeze(1), golden_tensor[:, di]) / self.args.max_length

        # Compute All Losses
        loss = loss_gen + loss_adv_style + loss_mul_style
        if self.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proj_style.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()  # Update learning rate schedule
            self.proj_style.zero_grad()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()
            self.decoder.zero_grad()
        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        losses_dic = {'loss_gen': loss_gen, '- loss_adv_style': loss_adv_style, 'loss_mul_style': loss_mul_style,
                      'loss_classifier_adv_style': loss_classifier_adv_style}
        return (losses_dic, decoded_words)

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def predict_style_vector(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        style = self.proj_style(encoder_hidden)
        return style

    def evaluate_for_style_transfert(self, input_tensor, style):
        loss_gen = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        # Sentence Generation
        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoder_hidden = encoder_hidden

        decoded_words = [decoder_input]

        # Without teacher forcing: use its own predictions as the next input
        for di in range(self.args.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
            decoded_words.append(topi.squeeze(-1))
            loss_gen += self.loss(decoder_output.squeeze(1), input_tensor[:, di]) / self.args.max_length

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        losses_dic = {'loss_gen': loss_gen}
        return (losses_dic, decoded_words)





class RenySeq2Seq(torch.nn.Module):
    def __init__(self, args, reny_dataloader):
        super(RenySeq2Seq, self).__init__()
        self.args = args
        self.training_all_except_encoder = True
        self.reny_dataloader = reny_dataloader
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.decoder = DecoderRNN(args, args.style_dim + args.content_dim, args.number_of_tokens, args.number_of_layers)
        self.loss = torch.nn.NLLLoss(ignore_index=self.args.tokenizer.pad_token_id)
        if self.args.complex_proj_content:
            self.proj_style = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                            nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                            nn.Linear(args.hidden_dim, args.style_dim))
            self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                              nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                              nn.Linear(args.hidden_dim, args.content_dim))
        else:
            self.proj_style = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                            nn.Linear(args.hidden_dim, args.style_dim))
            self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                              nn.Linear(args.hidden_dim, args.content_dim))

        self.args.ema_beta = args.ema_beta
        if not self.args.not_use_ema:
            self.ema_dict = {'ema_encoder': EMA(self.encoder, self.args, self.args.ema_beta),
                             'ema_proj_s': EMA(self.proj_style, self.args, self.args.ema_beta),
                             'ema_decoder': EMA(self.decoder, self.args, self.args.ema_beta),
                             'ema_proj_c': EMA(self.proj_content, self.args, self.args.ema_beta)}

            for _, value in self.ema_dict.items():
                value.register()

        # D_\gamma
        self.d_gamma = ClassifierGamma(args.content_dim + args.number_of_styles,
                                       args.number_of_styles)

        # Style classifier
        self.style_classifier = Classifier(args.content_dim, args.number_of_styles)

        # Loss : paper multipliers
        self.mul_mi = self.args.mul_mi

    def forward(self, input_tensor, labels, teacher_ratio):
        if self.args.special_clement:
            return self.forward_clement(input_tensor, labels, teacher_ratio)
        else:
            return self.forward_non_clement(input_tensor, labels, teacher_ratio)

    def forward_clement(self, input_tensor, labels, teacher_ratio):
        ###############################
        # Update style classifier loss:
        ###############################
        loss_gen, loss_mi, reny, loss_gamma, loss_style_classifier, loss_gen_reny = torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)
        gradient_encoder, gradient_decoder, gradient_content_proj, gradient_style_proj, gradient_reny = torch.tensor(
            0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)

        # epoch_iterator = tqdm(self.reny_dataloader, desc="Reny Training")
        if self.training_all_except_encoder:
            for step, batch in enumerate(self.reny_dataloader):
                if step == self.args.reny_training + 1:
                    break
                inputs_reny = batch['line'].to(self.args.device)
                labels_reny = batch['label'].to(self.args.device)
                free_params(self.style_classifier)
                frozen_params(self.d_gamma)
                frozen_params(self.proj_style)
                frozen_params(self.proj_content)
                frozen_params(self.encoder)
                frozen_params(self.decoder)

                encoder_hidden = self.encoder.initHidden()
                encoder_output, encoder_hidden = self.encoder(inputs_reny, encoder_hidden)

                # Content space
                content = self.proj_content(encoder_hidden)
                style_content_pred = self.style_classifier(content)
                loss_style_classifier = self.loss(style_content_pred, labels_reny)
                if self.training:
                    loss_style_classifier.backward()
                    torch.nn.utils.clip_grad_norm_(self.style_classifier.parameters(), self.args.max_grad_norm)
                    self.args.optimizer.step()
                    self.style_classifier.zero_grad()

                ################
                # Update D_gamma
                ################
                loss_gamma = 0
                free_params(self.d_gamma)
                frozen_params(self.style_classifier)
                frozen_params(self.proj_style)
                frozen_params(self.proj_content)
                frozen_params(self.encoder)
                frozen_params(self.decoder)
                inputs_reny_dgamma = inputs_reny.clone()
                encoder_hidden = self.encoder.initHidden()
                encoder_output, encoder_hidden = self.encoder(inputs_reny_dgamma, encoder_hidden)

                # Content space
                content = self.proj_content(encoder_hidden)

                style_content_pred = self.style_classifier(content)
                label_content_pred = style_content_pred.topk(1, dim=-1)[-1]
                label_v = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
                    self.args.device)
                label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in labels.tolist()]).to(
                    self.args.device)
                u = torch.cat([content, label_u.unsqueeze(0).float().repeat(4, 1, 1)],
                              dim=-1)  # 4 is for bid + 2 layers
                v = torch.cat([content, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)

                d_gamma_content_pred_u = self.d_gamma(u)  # - log
                d_gamma_content_pred_v = self.d_gamma(v)  # - log

                loss_gamma -= torch.mean(d_gamma_content_pred_u[:, 0]) / 2  # first term in (16)
                loss_gamma -= torch.mean(d_gamma_content_pred_v[:, 1]) / 2
                if self.training:
                    loss_gamma.backward()
                    gradient_reny += comput_gradient_norm(self.d_gamma)
                    torch.nn.utils.clip_grad_norm_(self.d_gamma.parameters(), self.args.max_grad_norm)
                    self.args.optimizer.step()
                    self.d_gamma.zero_grad()

                ################
                # Update Decoder
                ################
                free_params(self.decoder)
                frozen_params(self.d_gamma)
                frozen_params(self.style_classifier)
                frozen_params(self.proj_style)
                frozen_params(self.proj_content)
                frozen_params(self.encoder)

                input_tensor_reny_gen_golden = inputs_reny.clone()
                inputs_reny_gen = inputs_reny.clone()
                if self.args.add_noise:
                    inputs_reny_gen = corrupt_input(self,inputs_reny)

                encoder_hidden = self.encoder.initHidden()
                encoder_output, encoder_hidden = self.encoder(inputs_reny_gen, encoder_hidden)

                # Style space
                style = self.proj_style(encoder_hidden)
                # Content space
                content = self.proj_content(encoder_hidden)

                # Concatenate style and other
                encoder_hidden = torch.cat([style, content], dim=-1)

                # Sentence Generation
                loss_gen_reny = 0
                decoder_input = torch.ones(self.args.batch_size, 1).to(
                    self.args.device).long() * self.args.tokenizer.sep_token_id
                decoder_hidden = encoder_hidden
                use_teacher_forcing = True if random.random() < teacher_ratio else False

                decoded_words = [decoder_input]

                if use_teacher_forcing and self.training:
                    # Teacher forcing: Feed the target as the next input
                    for di in range(self.args.max_length):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        loss_gen_reny += self.loss(decoder_output.squeeze(1),
                                                   input_tensor_reny_gen_golden[:, di]) / self.args.max_length
                        decoder_input = input_tensor_reny_gen_golden[:, di].unsqueeze(-1)  # Teacher forcing

                else:
                    # Without teacher forcing: use its own predictions as the next input
                    for di in range(self.args.max_length):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
                        decoded_words.append(topi.squeeze(-1))
                        loss_gen_reny += self.loss(decoder_output.squeeze(1),
                                                   input_tensor_reny_gen_golden[:, di]) / self.args.max_length

                if self.training:
                    loss_gen_reny.backward()
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.max_grad_norm)
                    self.args.optimizer.step()
                    self.decoder.zero_grad()

        ###############################
        # Update Genloss + \lambda * MI
        ###############################
        input_tensor_gen_golden = input_tensor.clone()
        input_tensor_gen = input_tensor.clone()
        if self.args.add_noise:
            input_tensor_gen = corrupt_input(self,input_tensor_gen)
        input_tensor_mi = input_tensor_gen.clone()
        if self.training:
            frozen_params(self.d_gamma)
            frozen_params(self.style_classifier)
            frozen_params(self.decoder)
            free_params(self.proj_style)
            free_params(self.proj_content)
            free_params(self.encoder)

        # H(S)
        encoder_hidden_mi = self.encoder.initHidden()
        encoder_output_mi, encoder_hidden_mi = self.encoder(input_tensor_mi, encoder_hidden_mi)

        # Content space
        content_mi = self.proj_content(encoder_hidden_mi)
        style_content_pred_mi = self.style_classifier(content_mi)
        if self.args.alternative_hs:  # bad idea according to experiments
            loss_h_s = - torch.log(torch.mean(torch.exp(style_content_pred_mi[:, 0]))) * (
                    1 - torch.mean(labels.float()).item()) - torch.log(
                torch.mean(torch.exp(style_content_pred_mi[:, 1]))) * torch.mean(labels.float()).item()  # we use Gibbs
        else:
            loss_h_s = - torch.mean(style_content_pred_mi[:, 0]) * (1 - torch.mean(labels.float()).item()) - torch.mean(
                style_content_pred_mi[:, 1]) * torch.mean(labels.float()).item()  # we use Jensen

        # Compute Reny
        label_content_pred = style_content_pred_mi.topk(1, dim=-1)[-1]
        label_v = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
            self.args.device)
        label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in labels.tolist()]).to(self.args.device)
        u = torch.cat([content_mi, label_u.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)
        v = torch.cat([content_mi, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)
        d_gamma_content_pred_u = self.d_gamma(u)
        d_gamma_content_pred_v = self.d_gamma(v)
        R = torch.mean((torch.exp(d_gamma_content_pred_u[:, 0]) / torch.exp(d_gamma_content_pred_v[:, 1])) ** (
                self.args.alpha - 1))

        # Remove biais from gradients
        reny = torch.abs(torch.log(R) / (self.args.alpha - 1))

        loss_h_sz = self.loss(style_content_pred_mi, labels)

        # MI
        if self.args.no_reny:
            loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz)
        else:
            loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz + reny)

        #########################
        ######## Genloss ########
        #########################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Style space
        style = self.proj_style(encoder_hidden)
        # Content space
        content = self.proj_content(encoder_hidden)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        # Sentence Generation
        loss_gen = 0
        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_ratio else False

        decoded_words = [decoder_input]

        if use_teacher_forcing and self.training:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss_gen += self.loss(decoder_output.squeeze(1), input_tensor_gen_golden[:, di]) / self.args.max_length
                decoder_input = input_tensor_gen_golden[:, di].unsqueeze(-1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
                decoded_words.append(topi.squeeze(-1))
                loss_gen += self.loss(decoder_output.squeeze(1), input_tensor_gen_golden[:, di]) / self.args.max_length

        # Compute All Losses for MI
        if self.training:
            if self.args.no_minimization_of_mi_training:
                loss = loss_gen
            else:
                loss = loss_gen + loss_mi
            loss.backward()
            gradient_encoder, gradient_decoder, gradient_content_proj, gradient_style_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.decoder), comput_gradient_norm(
                self.proj_content), comput_gradient_norm(self.proj_style)
            torch.nn.utils.clip_grad_norm_(self.proj_style.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            if not self.args.not_use_ema:
                for _, value in self.ema_dict.items():
                    value.update()
            self.args.scheduler.step()  # Update learning rate schedule
            self.proj_style.zero_grad()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        losses_dic = {'loss_gen': loss_gen, 'loss_mi': loss_mi, 'reny': reny, 'loss_gamma': loss_gamma,
                      'loss_h_sz': loss_h_sz, 'loss_h_s': loss_h_s, 'loss_style_classifier': loss_style_classifier,
                      'gradient_encoder': gradient_encoder, 'gradient_decoder': gradient_decoder,
                      'loss_gen_reny': loss_gen_reny,
                      'gradient_content_proj': gradient_content_proj, 'gradient_style_proj': gradient_style_proj,
                      'gradient_reny': gradient_reny / self.args.reny_training}
        return (losses_dic, decoded_words, input_tensor_gen)

    def forward_non_clement(self, input_tensor, labels, teacher_ratio):
        ###############################
        # Update style classifier loss:
        ###############################
        loss_gen, loss_mi, reny, loss_gamma, loss_style_classifier = torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)
        gradient_encoder, gradient_decoder, gradient_content_proj, gradient_style_proj, gradient_reny = torch.tensor(
            0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)

        # epoch_iterator = tqdm(self.reny_dataloader, desc="Reny Training")
        for step, batch in enumerate(self.reny_dataloader):
            if step == self.args.reny_training + 1:
                break
            inputs_reny = batch['line'].to(self.args.device)
            labels_reny = batch['label'].to(self.args.device)
            free_params(self.style_classifier)
            frozen_params(self.d_gamma)
            frozen_params(self.proj_style)
            frozen_params(self.proj_content)
            frozen_params(self.encoder)
            frozen_params(self.decoder)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny, encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)
            style_content_pred = self.style_classifier(content)
            loss_style_classifier = self.loss(style_content_pred, labels_reny)
            if self.training:
                loss_style_classifier.backward()
                torch.nn.utils.clip_grad_norm_(self.style_classifier.parameters(), self.args.max_grad_norm)
                self.args.optimizer.step()
                self.style_classifier.zero_grad()

            ################
            # Update D_gamma
            ################
            loss_gamma = 0
            free_params(self.d_gamma)
            frozen_params(self.style_classifier)
            frozen_params(self.proj_style)
            frozen_params(self.proj_content)
            frozen_params(self.encoder)
            frozen_params(self.decoder)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny, encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)

            style_content_pred = self.style_classifier(content)
            label_content_pred = style_content_pred.topk(1, dim=-1)[-1]
            label_v = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
                self.args.device)
            label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in labels.tolist()]).to(self.args.device)
            u = torch.cat([content, label_u.unsqueeze(0).float().repeat(4, 1, 1)],
                          dim=-1)  # 4 is for bid + 2 layers
            v = torch.cat([content, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)

            d_gamma_content_pred_u = self.d_gamma(u)  # - log
            d_gamma_content_pred_v = self.d_gamma(v)  # - log

            loss_gamma -= torch.mean(d_gamma_content_pred_u[:, 0]) / 2  # first term in (16)
            loss_gamma -= torch.mean(d_gamma_content_pred_v[:, 1]) / 2
            if self.training:
                loss_gamma.backward()
                gradient_reny += comput_gradient_norm(self.d_gamma)
                torch.nn.utils.clip_grad_norm_(self.d_gamma.parameters(), self.args.max_grad_norm)
                self.args.optimizer.step()
                self.d_gamma.zero_grad()
            # self.args.logger.info('***** Gamma Loss {}'.format(loss_gamma.item()))

        ###############################
        # Update Genloss + \lambda * MI
        ###############################
        input_tensor_gen_golden = input_tensor.clone()
        input_tensor_gen = input_tensor.clone()
        if self.args.add_noise:
            input_tensor_gen = corrupt_input(self,input_tensor)
        input_tensor_noisy = input_tensor_gen.clone()
        # input_tensor_gen = input_tensor.clone()
        if self.training:
            frozen_params(self.d_gamma)
            frozen_params(self.style_classifier)
            free_params(self.proj_style)
            free_params(self.proj_content)
            free_params(self.encoder)
            free_params(self.decoder)

        # H(S)
        encoder_hidden_mi = self.encoder.initHidden()
        encoder_output_mi, encoder_hidden_mi = self.encoder(input_tensor_noisy, encoder_hidden_mi)

        # Content space
        content_mi = self.proj_content(encoder_hidden_mi)
        style_content_pred_mi = self.style_classifier(content_mi)
        if self.args.alternative_hs:
            loss_h_s = - torch.log(torch.mean(torch.exp(style_content_pred_mi[:, 0]))) * (
                    1 - torch.mean(labels.float()).item()) - torch.log(
                torch.mean(torch.exp(style_content_pred_mi[:, 1]))) * torch.mean(labels.float()).item()  # we use Gibbs
        else:
            loss_h_s = - torch.mean(style_content_pred_mi[:, 0]) * (1 - torch.mean(labels.float()).item()) - torch.mean(
                style_content_pred_mi[:, 1]) * torch.mean(labels.float()).item()  # we use Jensen

        # Compute Reny
        label_content_pred = style_content_pred_mi.topk(1, dim=-1)[-1]
        label_v = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
            self.args.device)
        label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in labels.tolist()]).to(self.args.device)
        u = torch.cat([content_mi, label_u.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)
        v = torch.cat([content_mi, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)
        d_gamma_content_pred_u = self.d_gamma(u)
        d_gamma_content_pred_v = self.d_gamma(v)
        R = torch.mean((torch.exp(d_gamma_content_pred_u[:, 0]) / torch.exp(d_gamma_content_pred_v[:, 1])) ** (
                self.args.alpha - 1))

        # Remove biais from gradients
        reny = torch.log(R) / (self.args.alpha - 1)

        loss_h_sz = self.loss(style_content_pred_mi, labels)

        # MI
        if self.args.no_reny:
            loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz)
        else:
            loss_mi = self.args.mul_mi * (loss_h_s - loss_h_sz + reny)

        #########################
        ######## Genloss ########
        #########################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Style space
        style = self.proj_style(encoder_hidden)
        # Content space
        content = self.proj_content(encoder_hidden)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        # Sentence Generation
        loss_gen = 0
        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoder_hidden = encoder_hidden
        use_teacher_forcing = False if random.random() < teacher_ratio else False

        decoded_words = [decoder_input]

        if use_teacher_forcing and self.training:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss_gen += self.loss(decoder_output.squeeze(1), input_tensor_gen_golden[:, di]) / self.args.max_length
                decoder_input = input_tensor_gen_golden[:, di].unsqueeze(-1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
                decoded_words.append(topi.squeeze(-1))
                loss_gen += self.loss(decoder_output.squeeze(1), input_tensor_gen_golden[:, di]) / self.args.max_length

        # Compute All Losses
        if self.training:
            if self.args.no_minimization_of_mi_training:
                loss = loss_gen
            else:
                loss = loss_gen + loss_mi
            loss.backward()
            gradient_encoder, gradient_decoder, gradient_content_proj, gradient_style_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.decoder), comput_gradient_norm(
                self.proj_content), comput_gradient_norm(self.proj_style)
            torch.nn.utils.clip_grad_norm_(self.proj_style.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            if not self.args.not_use_ema:
                for _, value in self.ema_dict.items():
                    value.update()
            self.args.scheduler.step()  # Update learning rate schedule
            self.proj_style.zero_grad()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()
            self.decoder.zero_grad()

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        losses_dic = {'loss_gen': loss_gen, 'loss_mi': loss_mi, 'reny': reny, 'loss_gamma': loss_gamma,
                      'loss_h_sz': loss_h_sz, 'loss_h_s': loss_h_s, 'loss_style_classifier': loss_style_classifier,
                      'gradient_encoder': gradient_encoder, 'gradient_decoder': gradient_decoder,
                      'gradient_content_proj': gradient_content_proj, 'gradient_style_proj': gradient_style_proj,
                      'gradient_reny': gradient_reny / self.args.reny_training}
        return (losses_dic, decoded_words, input_tensor_noisy)


    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def predict_style_vector(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        style = self.proj_style(encoder_hidden)
        return style

    def forward_reconstruction(self, input_tensor):
        loss = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Style space
        style = self.proj_style(encoder_hidden)
        # Content space
        content = self.proj_content(encoder_hidden)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoded_words = [decoder_input]
        decoder_hidden = encoder_hidden
        for di in range(self.args.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
                decoded_words.append(topi.squeeze(-1))
                loss += self.loss(decoder_output.squeeze(1), input_tensor[:, di])
        loss = loss / self.args.max_length

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        return ({'generation_loss': loss}, decoded_words)

    def evaluate_for_style_transfert(self, input_tensor, style):
        loss_gen = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        # Sentence Generation
        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoder_hidden = encoder_hidden

        decoded_words = [decoder_input]

        # Without teacher forcing: use its own predictions as the next input
        for di in range(self.args.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
            decoded_words.append(topi.squeeze(-1))
            loss_gen += self.loss(decoder_output.squeeze(1), input_tensor[:, di]) / self.args.max_length

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        losses_dic = {'loss_gen': loss_gen}
        return (losses_dic, decoded_words)
