from model_utils import *
import pickle
import torch
from tqdm import tqdm
import numpy as np
import copy
from geomloss import SamplesLoss
from mi_estimators import *


class StyleEmdedding(torch.nn.Module):
    """
    AAAI 2019 model 2
    """

    def __init__(self, args, reny_dataloader):
        super(StyleEmdedding, self).__init__()
        self.args = args
        self.training_all_except_encoder = True
        self.reny_dataloader = reny_dataloader
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.decoder = DecoderRNN(args, args.style_dim + args.content_dim, args.number_of_tokens, args.number_of_layers)
        self.loss = torch.nn.NLLLoss(ignore_index=self.args.tokenizer.pad_token_id)
        self.loss_classif = torch.nn.NLLLoss()
        if self.args.complex_proj_content:
            self.proj_style = nn.Sequential(nn.Linear(1, args.style_dim), nn.LeakyReLU(),
                                            nn.Linear(args.style_dim, args.style_dim), nn.LeakyReLU(),  # emb labels
                                            nn.Linear(args.style_dim, args.style_dim))  # emb labels
            self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                              nn.Linear(args.hidden_dim, args.content_dim), nn.LeakyReLU(),
                                              nn.Linear(args.hidden_dim, args.content_dim))
        else:
            self.proj_style = nn.Sequential(nn.Linear(1, args.style_dim), nn.LeakyReLU(),
                                            nn.Linear(args.style_dim, args.style_dim))  # emb labels
            self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                              nn.Linear(args.hidden_dim, args.content_dim))

        # D_\gamma
        self.d_gamma = ClassifierGamma(args.content_dim + args.number_of_styles,
                                       args.number_of_styles)

        # Style classifier
        self.style_classifier = Classifier(args.content_dim, args.number_of_styles, args.use_complex_classifier)

        # Loss : paper multipliers
        self.mul_mi = self.args.mul_mi

    def forward(self, input_tensor, labels, teacher_ratio):
        ###############################
        # Update style classifier loss:
        ###############################
        loss_gen, loss_mi, reny, loss_gamma, loss_style_classifier, loss_gen_reny, loss_h_sz, loss_h_s, loss_gen_reny = torch.tensor(
            0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)
        gradient_encoder, gradient_decoder, gradient_content_proj, gradient_style_proj, gradient_reny = torch.tensor(
            0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)

        reny_pos_class, reny_neg_class, reny_one_labels = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)
        pos_class, neg_class, one_labels = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)
        count = 0
        # epoch_iterator = tqdm(self.reny_dataloader, desc="Reny Training")
        if self.training_all_except_encoder:
            for step, batch in enumerate(self.reny_dataloader):
                if count == self.args.reny_training + 1:
                    break
                count += 1
                inputs_reny = batch['line'].to(self.args.device)
                labels_reny = batch['label'].to(self.args.device)
                free_params(self.style_classifier)
                frozen_params(self.d_gamma)
                frozen_params(self.proj_style)
                frozen_params(self.proj_content)
                frozen_params(self.encoder)
                frozen_params(self.decoder)

                encoder_hidden = self.encoder.initHidden()
                encoder_output, encoder_hidden = self.encoder(inputs_reny.clone(), encoder_hidden)

                # Content space
                content = self.proj_content(encoder_hidden)
                style_content_pred = self.style_classifier(content)
                loss_style_classifier = self.loss_classif(style_content_pred, labels_reny)

                reny_pos_class = torch.sum(torch.exp(style_content_pred)[:, 1])
                reny_neg_class = torch.sum(torch.exp(style_content_pred)[:, 0])
                reny_one_labels = torch.sum(labels_reny)
                if self.training:
                    loss_style_classifier.backward()
                    torch.nn.utils.clip_grad_norm_(self.style_classifier.parameters(), self.args.max_grad_norm)
                    self.args.optimizer.step()
                    self.style_classifier.zero_grad()

                ################
                # Update D_gamma
                ################
                if not self.args.no_minimization_of_mi_training:
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
                    label_content_pred = style_content_pred.topk(1, dim=-1)[-1].squeeze(-1)

                    if self.args.use_complex_gamma_training:
                        raise NotImplementedError

                    else:
                        label_v = torch.tensor(
                            [[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
                            self.args.device)
                        label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in labels_reny.tolist()]).to(
                            self.args.device)

                        u = torch.cat([content, label_u.unsqueeze(0).float().repeat(4, 1, 1)],
                                      dim=-1)  # 4 is for bid + 2 layers
                        v = torch.cat([content, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)

                        d_gamma_content_pred_u = self.d_gamma(u)  # - log
                        d_gamma_content_pred_v = self.d_gamma(v)  # - log

                        loss_gamma = - torch.mean(d_gamma_content_pred_u[:, 0]) / 2 - torch.mean(
                            d_gamma_content_pred_v[:, 1]) / 2

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
                    inputs_reny_gen = corrupt_input(self, inputs_reny_gen)

                encoder_hidden = self.encoder.initHidden()
                encoder_output, encoder_hidden = self.encoder(inputs_reny_gen, encoder_hidden)

                # Style space
                style = self.proj_style(labels_reny.float().unsqueeze(1))
                style = torch.cat([style.unsqueeze(0) for _ in range(4)])
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
            input_tensor_gen = corrupt_input(self, input_tensor_gen)
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

        if not self.args.no_minimization_of_mi_training:
            len_ = len(labels.tolist())
            loss_h_s = (-1) * torch.log(torch.sum(torch.exp(style_content_pred_mi)[:, 0]) / len_) * (
                    1 - torch.sum(labels).item() / len_) - torch.log(
                torch.sum(torch.exp(style_content_pred_mi)[:, 1]) / len_) * (torch.sum(labels).item() / len_)
            pos_class = torch.sum(torch.exp(style_content_pred_mi)[:, 1])
            neg_class = torch.sum(torch.exp(style_content_pred_mi)[:, 0])
            one_labels = torch.sum(labels)
            # loss_h_s = - torch.mean(style_content_pred_mi[:, 0]) * (1 - torch.mean(labels.float()).item()) - torch.mean(
            # style_content_pred_mi[:, 1]) * torch.mean(labels.float()).item()  # we use Jensen
            # print(loss_h_s)
            # Compute Reny
            label_content_pred = style_content_pred_mi.topk(1, dim=-1)[-1].squeeze(-1)
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

            loss_h_sz = self.loss_classif(style_content_pred_mi, labels)

            # MI
            if self.args.no_reny:
                loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz)
            else:
                loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz + reny)
        elif self.args.use_club_estimation:
            log_p_y_true_given_x = torch.masked_select(self.style_classifier(content_mi),
                                                       torch.nn.functional.one_hot(labels).bool())
            log_p_y_false_given_x = torch.masked_select(self.style_classifier(content_mi),
                                                        ~(torch.nn.functional.one_hot(labels).bool()))
            loss_mi = - self.args.mul_mi * torch.mean(log_p_y_true_given_x - log_p_y_false_given_x)
        else:
            loss_mi = - self.args.mul_mi * self.loss_classif(style_content_pred_mi, labels)

        #########################
        ######## Genloss ########
        #########################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Style space
        style = self.proj_style(labels.float().unsqueeze(1))
        style = torch.cat([style.unsqueeze(0) for _ in range(4)])
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
            loss = torch.abs(loss_gen + loss_mi)
            loss.backward()
            gradient_encoder, gradient_decoder, gradient_content_proj, gradient_style_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.decoder), comput_gradient_norm(
                self.proj_content), comput_gradient_norm(self.proj_style)
            torch.nn.utils.clip_grad_norm_(self.proj_style.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()
            self.proj_style.zero_grad()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()

        decoded_words = torch.cat(decoded_words, dim=-1)  # memory ineficient but nicer
        losses_dic = {'loss_gen': loss_gen, 'loss_mi': loss_mi, 'reny': reny, 'loss_gamma': loss_gamma,
                      'loss_h_sz': loss_h_sz, 'loss_h_s': loss_h_s, 'loss_style_classifier': loss_style_classifier,
                      'gradient_encoder': gradient_encoder, 'gradient_decoder': gradient_decoder,
                      'loss_gen_reny': loss_gen_reny, 'pos_class': pos_class, 'neg_class': neg_class,
                      'one_labels': one_labels,
                      'reny_pos_class': reny_pos_class, 'reny_neg_class': reny_neg_class,
                      'reny_one_labels': reny_one_labels,
                      'gradient_content_proj': gradient_content_proj, 'gradient_style_proj': gradient_style_proj,
                      'gradient_reny': gradient_reny / self.args.reny_training}
        return (losses_dic, decoded_words, input_tensor_gen)

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def forward_transfert(self, inputs, labels_to_transfert, pos_style, neg_style):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)
        # Style space
        style = self.proj_style(labels_to_transfert.float().unsqueeze(1))
        style = torch.cat([style.unsqueeze(0) for _ in range(4)])

        # Content space
        content = self.proj_content(encoder_hidden)

        # Concatenate style and other
        encoder_hidden = torch.cat([style, content], dim=-1)

        # Sentence Generation
        loss_gen_reny = 0
        decoder_input = torch.ones(self.args.batch_size, 1).to(
            self.args.device).long() * self.args.tokenizer.sep_token_id
        decoder_hidden = encoder_hidden

        decoded_words = [decoder_input]

        for di in range(self.args.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(-1)  # detach from history as input
            decoded_words.append(topi.squeeze(-1))
        decoded_words = torch.cat(decoded_words, dim=-1)  # clean data
        return decoded_words


class RaoClassificationStyleEmdedding(torch.nn.Module):
    """
    AAAI 2019 model 2
    """

    def __init__(self, args, reny_dataloader):
        super(RaoClassificationStyleEmdedding, self).__init__()
        self.args = args
        self.reny_dataloader = reny_dataloader
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.loss_classif = torch.nn.NLLLoss()

        self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                          nn.Linear(args.hidden_dim, args.content_dim))

        # D_\gamma
        self.d_gamma = ClassifierGamma(args.content_dim + args.number_of_styles,
                                       args.number_of_styles)

        self.advers_classifier = Classifier(args.content_dim, args.number_of_styles, args.use_complex_classifier)
        self.downstream_classifier = Classifier(args.content_dim, args.number_of_styles,
                                                args.use_complex_classifier)

        # Loss : paper multipliers
        self.mul_mi = self.args.mul_mi

    def js_loss(self, mean_0, mean_1, var_0, var_1):
        """
        1/2[log|Σ2|/|Σ1|−𝑑+tr{Σ**0.5Σ1}+(𝜇2−𝜇1)𝑇Σ−12(𝜇2−𝜇1)]
        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        """
        d = var_1.size(1)
        var_0 = torch.diag(var_0)
        var_1 = torch.diag(var_1)
        log_det_0_det_1 = (torch.sum(torch.log(var_0), dim=0) - torch.sum(torch.log(var_1), dim=0))
        log_det_1_det_0 = (torch.sum(torch.log(var_1), dim=0) - torch.sum(torch.log(var_0), dim=0))
        tr_0_1 = torch.sum(var_0 / var_1)
        tr_1_0 = torch.sum(var_1 / var_0)
        last_1 = torch.matmul((mean_0 - mean_1) * (var_1 ** (-1)), mean_0 - mean_1)
        last_0 = torch.matmul((mean_0 - mean_1) * (var_0 ** (-1)), mean_0 - mean_1)

        js = -2 * d + (log_det_0_det_1 + tr_1_0 + last_1 + log_det_1_det_0 + tr_0_1 + last_0)
        return js / 4

    def rao_loss(self, mean_0, mean_1, var_0, var_1):
        """
        https://www.sciencedirect.com/science/article/pii/S0166218X14004211
        """
        first = (((mean_0 - mean_1) ** 2) / 2 + (torch.diag(var_1) + torch.diag(var_0)) ** 2) ** (1 / 2)
        second = (((mean_0 - mean_1) ** 2) / 2 + (torch.diag(var_1) - torch.diag(var_0)) ** 2) ** (1 / 2)
        rao = torch.sqrt(torch.sum((torch.log((first + second) / (first - second))) ** 2) * 2)
        return rao

    def _matrix_pow(self, matrix, p=1 / 2):
        r"""
        Power of a matrix using Eigen Decomposition.
        Args:
            matrix: matrix
            p: power
        Returns:
            Power of a matrix
        """
        vals, vecs = torch.eig(matrix, eigenvectors=True)
        vals = torch.view_as_complex(vals.contiguous())
        vals_pow = vals.pow(p)
        vals_pow = torch.view_as_real(vals_pow)[:, 0]
        matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
        return matrix_pow

    def frechet_loss(self, mean_0, mean_1, var_0, var_1):
        var_0 = torch.diag(var_0)
        var_1 = torch.diag(var_1)
        return torch.norm(mean_0 - mean_1, p=2) ** 2 + torch.sum(var_0 + var_1 - 2 * (var_0 * var_1) ** (1 / 2))

    def compute_distance(self, ones_content, zeros_content):
        mu_one = torch.mean(ones_content, dim=0)
        sigma_ones = torch.matmul((ones_content - mu_one).transpose(1, 0), ones_content - mu_one)
        sigma_ones[torch.eye(sigma_ones.size(0)) == 0] = 0

        mu_zeros = torch.mean(zeros_content, dim=0)
        sigma_zeros = torch.matmul((zeros_content - mu_zeros).transpose(1, 0), zeros_content - mu_zeros)
        sigma_zeros[torch.eye(sigma_zeros.size(0)) == 0] = 0
        if self.args.loss_type == 'frechet':
            return self.frechet_loss(mu_one, mu_zeros, sigma_ones, sigma_zeros)
        elif self.args.loss_type == 'rao':
            return self.rao_loss(mu_one, mu_zeros, sigma_ones, sigma_zeros)
        elif self.args.loss_type == 'js':
            return self.js_loss(mu_one, mu_zeros, sigma_ones, sigma_zeros)
        elif self.args.loss_type == 'wasserstein':
            loss = SamplesLoss(loss=self.args.loss_wasserstein, p=self.args.power, blur=self.args.blur)
            return loss(ones_content, zeros_content)

        else:
            raise NotImplementedError

    def forward(self, input_tensor, protected_labels, downstream_labels):
        loss_mi, reny, loss_gamma, loss_style_classifier, loss_h_sz, loss_h_s, loss_adv_classifier = torch.tensor(
            0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)
        gradient_encoder, gradient_content_proj, gradient_reny = torch.tensor(0.0).to(self.args.device), torch.tensor(
            0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)

        reny_pos_class, reny_neg_class, reny_one_labels = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)
        pos_class, neg_class, one_labels = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)

        count = 0

        ###############################
        # Update Genloss + \lambda * MI
        ###############################
        input_tensor_gen = input_tensor.clone()
        if self.args.add_noise:
            input_tensor_gen = corrupt_input(self, input_tensor_gen)
        ###################
        # RAO REGULARIZER #
        ###################
        ones = []
        zeros = []
        for step, batch in enumerate(self.reny_dataloader):
            #
            if count == self.args.reny_training + 1:
                break
            count += 1
            inputs_reny = batch['line'].to(self.args.device)
            labels_reny_downstream = batch['downstream_labels'].to(self.args.device)
            labels_reny_protected = batch['label'].to(self.args.device)
            encoder_hidden_reny = self.encoder.initHidden()
            encoder_output_reny, encoder_hidden_reny = self.encoder(inputs_reny, encoder_hidden_reny)

            content = torch.sum(self.proj_content(encoder_hidden_reny), dim=0)
            one_labels_content = content[labels_reny_protected == 1]
            ones.append(one_labels_content)
            zero_labels_content = content[labels_reny_protected == 0]
            zeros.append(zero_labels_content)
        ones_content = torch.cat(ones)
        zeros_content = torch.cat(zeros)
        loss_mi = self.args.mul_mi * self.compute_distance(ones_content, zeros_content)

        ###################
        # Downstream loss #
        ###################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_pred = self.downstream_classifier(content)
        loss_down_classifier = self.loss_classif(down_pred, downstream_labels)

        # Compute All Losses for MI
        if self.training:
            loss = torch.abs(loss_down_classifier + loss_mi)
            loss.backward()
            gradient_encoder, gradient_content_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.proj_content)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()

        losses_dic = {'loss_down_classifier': loss_down_classifier,
                      'loss_mi': loss_mi, 'reny': reny,
                      'loss_gamma': loss_gamma,
                      'loss_h_sz': loss_h_sz,
                      'loss_h_s': loss_h_s,
                      'loss_style_classifier': loss_style_classifier,
                      'gradient_encoder': gradient_encoder,
                      'pos_class': pos_class,
                      'neg_class': neg_class,
                      'one_labels': one_labels,
                      'reny_pos_class': reny_pos_class,
                      'reny_neg_class': reny_neg_class,
                      'reny_one_labels': reny_one_labels,
                      'gradient_content_proj': gradient_content_proj,
                      'loss_adv_classifier': loss_adv_classifier,
                      'gradient_reny': gradient_reny / self.args.reny_training}
        return losses_dic,

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def predict_downstream(self, inputs):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_preds = self.downstream_classifier(content)

        return down_preds


class ClassificationStyleEmdedding(torch.nn.Module):
    """
    AAAI 2019 model 2
    """

    def __init__(self, args, reny_dataloader):
        super(ClassificationStyleEmdedding, self).__init__()
        self.args = args
        self.reny_dataloader = reny_dataloader
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.loss_classif = torch.nn.NLLLoss()

        self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                          nn.Linear(args.hidden_dim, args.content_dim))

        # D_\gamma
        self.d_gamma = ClassifierGamma(args.content_dim + args.number_of_styles,
                                       args.number_of_styles)

        # TODO : Classifier : here we assume binary for both cases
        self.advers_classifier = Classifier(args.content_dim, args.number_of_styles, args.use_complex_classifier)
        self.downstream_classifier = Classifier(args.content_dim, args.number_of_styles,  # number_of_downstream_labels
                                                args.use_complex_classifier)

        # Loss : paper multipliers
        self.mul_mi = self.args.mul_mi

    def forward(self, input_tensor, protected_labels, downstream_labels):
        loss_mi, reny, loss_gamma, loss_style_classifier, loss_h_sz, loss_h_s, loss_adv_classifier = torch.tensor(
            0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)
        gradient_encoder, gradient_content_proj, gradient_reny = torch.tensor(0.0).to(self.args.device), torch.tensor(
            0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)

        reny_pos_class, reny_neg_class, reny_one_labels = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)
        pos_class, neg_class, one_labels = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device), torch.tensor(0.0).to(self.args.device)

        count = 0
        for step, batch in enumerate(self.reny_dataloader):
            if count == self.args.reny_training + 1:
                break
            count += 1
            # TODO : make both adv and downstream labels
            inputs_reny = batch['line'].to(self.args.device)
            labels_reny_downstream = batch['downstream_labels'].to(self.args.device)
            labels_reny_protected = batch['label'].to(self.args.device)
            free_params(self.advers_classifier)
            frozen_params(self.downstream_classifier)
            frozen_params(self.d_gamma)
            frozen_params(self.proj_content)
            frozen_params(self.encoder)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny.clone(), encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)
            adv_pred = self.advers_classifier(content)
            loss_adv_classifier = self.loss_classif(adv_pred, labels_reny_protected)

            reny_pos_class = torch.sum(torch.exp(adv_pred)[:, 1])
            reny_neg_class = torch.sum(torch.exp(adv_pred)[:, 0])
            reny_one_labels = torch.sum(labels_reny_protected)
            if self.training:
                loss_adv_classifier.backward()
                torch.nn.utils.clip_grad_norm_(self.advers_classifier.parameters(), self.args.max_grad_norm)
                self.args.optimizer.step()
                self.advers_classifier.zero_grad()

            ################
            # Update D_gamma
            ################
            if not self.args.no_minimization_of_mi_training:
                loss_gamma = 0
                free_params(self.d_gamma)
                frozen_params(self.downstream_classifier)
                frozen_params(self.advers_classifier)
                frozen_params(self.proj_content)
                frozen_params(self.encoder)

                inputs_reny_dgamma = inputs_reny.clone()
                encoder_hidden = self.encoder.initHidden()
                encoder_output, encoder_hidden = self.encoder(inputs_reny_dgamma, encoder_hidden)

                # Content space
                content = self.proj_content(encoder_hidden)

                adv_content_pred = self.advers_classifier(content)
                label_content_pred = adv_content_pred.topk(1, dim=-1)[-1].squeeze(-1)

                label_v = torch.tensor(
                    [[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
                    self.args.device)
                label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in labels_reny_protected.tolist()]).to(
                    self.args.device)

                u = torch.cat([content, label_u.unsqueeze(0).float().repeat(4, 1, 1)],
                              dim=-1)  # 4 is for bid + 2 layers
                v = torch.cat([content, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)

                d_gamma_content_pred_u = self.d_gamma(u)  # - log
                d_gamma_content_pred_v = self.d_gamma(v)  # - log # TODO :)

                loss_gamma = - torch.mean(d_gamma_content_pred_u[:, 0]) / 2 - torch.mean(
                    d_gamma_content_pred_v[:, 1]) / 2

                if self.training:
                    loss_gamma.backward()
                    gradient_reny += comput_gradient_norm(self.d_gamma)
                    torch.nn.utils.clip_grad_norm_(self.d_gamma.parameters(), self.args.max_grad_norm)
                    self.args.optimizer.step()
                    self.d_gamma.zero_grad()

            ##########################
            # Update downstream taks #
            ##########################
            free_params(self.downstream_classifier)
            frozen_params(self.d_gamma)
            frozen_params(self.advers_classifier)
            frozen_params(self.proj_content)
            frozen_params(self.encoder)

            inputs_reny_gen = inputs_reny.clone()
            if self.args.add_noise:
                inputs_reny_gen = corrupt_input(self, inputs_reny_gen)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny_gen, encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)
            down_pred = self.downstream_classifier(content)
            loss_down_classifier = self.loss_classif(down_pred, labels_reny_downstream)

            if self.training:
                loss_down_classifier.backward()
                torch.nn.utils.clip_grad_norm_(self.downstream_classifier.parameters(), self.args.max_grad_norm)
                self.args.optimizer.step()
                self.downstream_classifier.zero_grad()

        ###############################
        # Update Genloss + \lambda * MI
        ###############################
        input_tensor_gen = input_tensor.clone()
        if self.args.add_noise:
            input_tensor_gen = corrupt_input(self, input_tensor_gen)
        input_tensor_mi = input_tensor_gen.clone()
        if self.training:
            frozen_params(self.d_gamma)
            frozen_params(self.advers_classifier)
            frozen_params(self.downstream_classifier)
            free_params(self.proj_content)
            free_params(self.encoder)

        # H(S)
        encoder_hidden_mi = self.encoder.initHidden()
        encoder_output_mi, encoder_hidden_mi = self.encoder(input_tensor_mi, encoder_hidden_mi)

        # Content space
        content_mi = self.proj_content(encoder_hidden_mi)
        adv_content_pred_mi = self.advers_classifier(content_mi)

        if not self.args.no_minimization_of_mi_training:
            len_ = len(protected_labels.tolist())
            loss_h_s = (-1) * torch.log(torch.sum(torch.exp(adv_content_pred_mi)[:, 0]) / len_) * (
                    1 - torch.sum(protected_labels).item() / len_) - torch.log(
                torch.sum(torch.exp(adv_content_pred_mi)[:, 1]) / len_) * (torch.sum(protected_labels).item() / len_)
            pos_class = torch.sum(torch.exp(adv_content_pred_mi)[:, 1])
            neg_class = torch.sum(torch.exp(adv_content_pred_mi)[:, 0])
            one_labels = torch.sum(protected_labels)

            # Compute Reny
            label_content_pred = adv_content_pred_mi.topk(1, dim=-1)[-1].squeeze(-1)  # TODO ?
            label_v = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in label_content_pred.tolist()]).to(
                self.args.device)
            label_u = torch.tensor([[1., 0.] if el == 0 else [0., 1.] for el in protected_labels.tolist()]).to(
                self.args.device)
            u = torch.cat([content_mi, label_u.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)
            v = torch.cat([content_mi, label_v.unsqueeze(0).float().repeat(4, 1, 1)], dim=-1)
            d_gamma_content_pred_u = self.d_gamma(u)
            d_gamma_content_pred_v = self.d_gamma(v)
            R = torch.mean((torch.exp(d_gamma_content_pred_u[:, 0]) / torch.exp(d_gamma_content_pred_v[:, 1])) ** (
                    self.args.alpha - 1))

            # Remove biais from gradients
            reny = torch.abs(torch.log(R) / (self.args.alpha - 1))

            loss_h_sz = self.loss_classif(adv_content_pred_mi, protected_labels)

            # MI
            if self.args.no_reny:
                loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz)
            else:
                loss_mi = self.args.mul_mi * torch.abs(loss_h_s - loss_h_sz + reny)
        elif self.args.use_club_estimation:
            log_p_y_true_given_x = torch.masked_select(self.advers_classifier(content_mi),
                                                       torch.nn.functional.one_hot(protected_labels).bool())
            log_p_y_false_given_x = torch.masked_select(self.advers_classifier(content_mi),
                                                        ~(torch.nn.functional.one_hot(protected_labels).bool()))
            loss_mi = self.args.mul_mi * torch.mean(log_p_y_true_given_x - log_p_y_false_given_x)
        else:
            loss_mi = - self.args.mul_mi * self.loss_classif(adv_content_pred_mi, protected_labels)

        ###################
        # Downstream loss #
        ###################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_pred = self.downstream_classifier(content)
        loss_down_classifier = self.loss_classif(down_pred, downstream_labels)

        # Compute All Losses for MI
        if self.training:
            loss = torch.abs(loss_down_classifier + loss_mi)
            loss.backward()
            gradient_encoder, gradient_content_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.proj_content)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()

        losses_dic = {'loss_down_classifier': loss_down_classifier,
                      'loss_mi': loss_mi, 'reny': reny,
                      'loss_gamma': loss_gamma,
                      'loss_h_sz': loss_h_sz,
                      'loss_h_s': loss_h_s,
                      'loss_style_classifier': loss_style_classifier,
                      'gradient_encoder': gradient_encoder,
                      'pos_class': pos_class,
                      'neg_class': neg_class,
                      'one_labels': one_labels,
                      'reny_pos_class': reny_pos_class,
                      'reny_neg_class': reny_neg_class,
                      'reny_one_labels': reny_one_labels,
                      'gradient_content_proj': gradient_content_proj,
                      'loss_adv_classifier': loss_adv_classifier,
                      'gradient_reny': gradient_reny / self.args.reny_training}
        return losses_dic,

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def predict_downstream(self, inputs):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_preds = self.downstream_classifier(content)

        return down_preds


class KernelClassificationStyleEmdedding(torch.nn.Module):
    """
    AAAI 2019 model 2
    """

    def __init__(self, args, kernel_dataloader):
        super(KernelClassificationStyleEmdedding, self).__init__()
        self.args = args
        self.kernel_dataloader = kernel_dataloader
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.loss_classif = torch.nn.NLLLoss()

        self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                          nn.Linear(args.hidden_dim, args.content_dim))

        # D_\gamma
        self.d_gamma = ClassifierGamma(args.content_dim + args.number_of_styles,
                                       args.number_of_styles)

        self.number_of_point_kernel = args.kernel_size
        if self.args.mi_estimator == 'KNIFE':
            self.conditionnal_entropy_kernel_1 = MultiGaussKernelEE(args, self.number_of_point_kernel,
                                                                    args.content_dim,
                                                                    # [K, d] to initialize the kernel :) so K is the number of points :)
                                                                    average='weighted',  # un
                                                                    cov_diagonal='var',
                                                                    # diagonal of the covariance
                                                                    cov_off_diagonal='var')  # var)
            self.conditionnal_entropy_kernel_2 = MultiGaussKernelEE(args, self.number_of_point_kernel,
                                                                    args.content_dim,
                                                                    # [K, d] to initialize the kernel :) so K is the number of points :)
                                                                    average='weighted',  # un
                                                                    cov_diagonal='var',
                                                                    # diagonal of the covariance
                                                                    cov_off_diagonal='var')  # var)

            self.entropy_kernel = MultiGaussKernelEE(args, self.number_of_point_kernel * args.number_of_styles,
                                                     args.content_dim,
                                                     # [K, d] to initialize the kernel :) so K is the number of points :)
                                                     average='weighted',  # un
                                                     cov_diagonal='var',  # diagonal of the covariance
                                                     cov_off_diagonal='var')  # var)
        else:
            assert self.args.mi_estimator == 'DOE'
            self.conditionnal_entropy_kernel_1 = PDF(args.content_dim, "gauss")

            self.conditionnal_entropy_kernel_2 = PDF(args.content_dim, "gauss")

            self.entropy_kernel = PDF(args.content_dim, "gauss")

        self.advers_classifier = Classifier(args.content_dim, args.number_of_styles, args.use_complex_classifier)
        self.downstream_classifier = Classifier(args.content_dim, args.number_of_styles,
                                                # args.number_of_downstream_labels
                                                args.use_complex_classifier)

        # Loss : paper multipliers
        self.mul_mi = self.args.mul_mi

    def forward(self, input_tensor, protected_labels, downstream_labels):
        count = 0
        for step, batch in enumerate(self.kernel_dataloader):
            if count == self.args.reny_training + 1:
                break
            count += 1
            inputs_reny = batch['line'].to(self.args.device)
            labels_reny_downstream = batch['downstream_labels'].to(self.args.device)
            labels_reny_protected = batch['label'].to(self.args.device)

            ###################################
            ##### Compute h(Z|Y=y) & h(Z) #####
            ###################################
            frozen_params(self.entropy_kernel)
            frozen_params(self.proj_content)
            free_params(self.conditionnal_entropy_kernel_1)
            frozen_params(self.conditionnal_entropy_kernel_2)
            frozen_params(self.encoder)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny.clone(), encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)

            loss_entropy_conditionnal_0 = self.conditionnal_entropy_kernel_1(
                torch.sum(content, dim=0)[(labels_reny_protected == 0).bool()]) * (torch.sum(
                (labels_reny_protected.long() == 0).bool()) / float(labels_reny_protected.size(
                0)))  # is there a scale pb ? are we sure we output h(Z|y)
            if self.training:
                loss_entropy_conditionnal_0.backward()
                torch.nn.utils.clip_grad_norm_(self.conditionnal_entropy_kernel_1.parameters(),
                                               self.args.max_grad_norm)
                self.args.optimizer.step()
                self.conditionnal_entropy_kernel_1.zero_grad()

            frozen_params(self.conditionnal_entropy_kernel_1)
            free_params(self.conditionnal_entropy_kernel_2)
            loss_entropy_conditionnal_1 = self.conditionnal_entropy_kernel_2(
                torch.sum(content, dim=0)[(labels_reny_protected == 1).bool()]) * (torch.sum(
                (labels_reny_protected.long() == 1).bool()) / float(labels_reny_protected.size(
                0))).item()  # is there a scale pb ? are we sure we output h(Z|y)
            if self.training:
                loss_entropy_conditionnal_1.backward()
                torch.nn.utils.clip_grad_norm_(self.conditionnal_entropy_kernel_2.parameters(),
                                               self.args.max_grad_norm)
                self.args.optimizer.step()
                self.conditionnal_entropy_kernel_2.zero_grad()

            free_params(self.entropy_kernel)

            loss_entropy_conditionnal_full = self.entropy_kernel(
                torch.sum(content, dim=0))  # is there a scale pb ? are we sure we output h(Z|y)
            if self.training and not self.args.tight_training:
                loss_entropy_conditionnal_full.backward()
                torch.nn.utils.clip_grad_norm_(self.entropy_kernel.parameters(),
                                               self.args.max_grad_norm)
                self.args.optimizer.step()
                self.entropy_kernel.zero_grad()
            frozen_params(self.entropy_kernel)

            ##########################
            # Update downstream taks #
            ##########################
            free_params(self.downstream_classifier)
            frozen_params(self.entropy_kernel)
            frozen_params(self.proj_content)
            frozen_params(self.conditionnal_entropy_kernel_1)
            frozen_params(self.conditionnal_entropy_kernel_2)
            frozen_params(self.encoder)

            inputs_reny_gen = inputs_reny.clone()
            if self.args.add_noise:
                inputs_reny_gen = corrupt_input(self, inputs_reny_gen)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny_gen, encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)
            down_pred = self.downstream_classifier(content)
            loss_down_classifier = self.loss_classif(down_pred, labels_reny_downstream)

            if self.training:
                loss_down_classifier.backward()
                torch.nn.utils.clip_grad_norm_(self.downstream_classifier.parameters(), self.args.max_grad_norm)
                self.args.optimizer.step()
                self.downstream_classifier.zero_grad()

        if self.training and self.args.tight_training:
            self.entropy_kernel.update_parameters(
                {1: self.conditionnal_entropy_kernel_1, 2: self.conditionnal_entropy_kernel_2})
            if self.args.fine_tune_hkernel:
                for step, batch in enumerate(self.kernel_dataloader):
                    if count == self.args.reny_training + 1:
                        break
                    count += 1
                    inputs_reny = batch['line'].to(self.args.device)

                    #########################
                    ##### Compute  h(Z) #####
                    #########################
                    free_params(self.entropy_kernel)
                    frozen_params(self.downstream_classifier)
                    frozen_params(self.proj_content)
                    frozen_params(self.conditionnal_entropy_kernel_1)
                    frozen_params(self.conditionnal_entropy_kernel_2)
                    frozen_params(self.encoder)

                    encoder_hidden = self.encoder.initHidden()
                    encoder_output, encoder_hidden = self.encoder(inputs_reny.clone(), encoder_hidden)

                    # Content space
                    content = self.proj_content(encoder_hidden)
                    loss_entropy = self.entropy_kernel(torch.sum(content, dim=0))

                    if self.training:
                        loss_entropy.backward()
                        torch.nn.utils.clip_grad_norm_(self.entropy_kernel.parameters(), self.args.max_grad_norm)
                        self.args.optimizer.step()
                        self.entropy_kernel.zero_grad()

        ###############################
        # Update Genloss + \lambda * MI
        ###############################
        input_tensor_gen = input_tensor.clone()
        if self.args.add_noise:
            input_tensor_gen = corrupt_input(self, input_tensor_gen)
        input_tensor_mi = input_tensor_gen.clone()
        if self.training:
            frozen_params(self.downstream_classifier)
            frozen_params(self.entropy_kernel)
            free_params(self.proj_content)
            frozen_params(self.conditionnal_entropy_kernel_1)
            frozen_params(self.conditionnal_entropy_kernel_2)
            free_params(self.encoder)

        # Compute MI with protected_labels
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_mi, encoder_hidden)
        content = self.proj_content(encoder_hidden)

        h_maginal_1 = self.conditionnal_entropy_kernel_1(
            torch.sum(content, dim=0)[(protected_labels == 0).bool()]) * (torch.sum(
            (protected_labels.long() == 0).bool()) / float(protected_labels.size(
            0))).item()  # is there a scale pb ? are we sure we output h(Z|y)
        # un truc de fou furieux le float? wtf

        h_maginal_2 = self.conditionnal_entropy_kernel_2(
            torch.sum(content, dim=0)[(protected_labels == 1).bool()]) * (torch.sum(
            (protected_labels.long() == 1).bool()) / float(protected_labels.size(
            0))).item()  # is there a scale pb ? are we sure we output h(Z|y)

        hz = self.entropy_kernel(torch.sum(content, dim=0))
        loss_mi = torch.abs(hz - h_maginal_1 - h_maginal_2)

        ###################
        # Downstream loss #
        ###################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_pred = self.downstream_classifier(content)
        loss_down_classifier = self.loss_classif(down_pred, downstream_labels)

        # Compute All Losses for MI
        gradient_encoder, gradient_content_proj = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device)
        if self.training:
            loss = loss_down_classifier + self.args.mul_mi * loss_mi
            loss.backward()
            gradient_encoder, gradient_content_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.proj_content)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()

        losses_dic = {'loss_down_classifier': loss_down_classifier,
                      'loss_mi': loss_mi,
                      'gradient_encoder': gradient_encoder,
                      'hz': hz,
                      'h_maginal': h_maginal_1 + h_maginal_2,
                      'h_maginal_1': h_maginal_1,
                      'h_maginal_2': h_maginal_2,
                      'gradient_content_proj': gradient_content_proj}
        return losses_dic,

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def predict_downstream(self, inputs):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_preds = self.downstream_classifier(content)

        return down_preds


class MultiGaussKernelEE(nn.Module):
    def __init__(self, args, number_of_samples, hidden_size,
                 # [K, d] to initialize the kernel :) so K is the number of points :)
                 average='weighted',  # un
                 cov_diagonal='var',  # diagonal of the covariance
                 cov_off_diagonal='var',  # var
                 ):

        self.K, self.d = number_of_samples, hidden_size
        super(MultiGaussKernelEE, self).__init__()
        self.args = args

        # base_samples.requires_grad = False
        # if kernel_positions in ('fixed', 'free'):
        #    self.mean = base_samples[None, :, :].to(self.args.device)
        # else:
        #    self.mean = base_samples[None, None, :, :].to(self.args.device)  # [1, 1, K, d]

        # self.K = K
        # self.d = d

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(
            self.args.device)

        self.means = nn.Parameter(0.001 * torch.rand(number_of_samples, hidden_size), requires_grad=True).to(
            # 0 for > 0.001
            self.args.device)
        if cov_diagonal == 'const':
            diag = torch.ones((1, 1, self.d))
        elif cov_diagonal == 'var':
            diag = torch.ones((1, self.K, self.d))
        else:
            assert False, f'Invalid cov_diagonal: {cov_diagonal}'
        self.diag = nn.Parameter(diag.to(self.args.device))

        if cov_off_diagonal == 'var':
            tri = torch.zeros((1, self.K, self.d, self.d))
            self.tri = nn.Parameter(tri.to(self.args.device))
        elif cov_off_diagonal == 'zero':
            self.tri = None
        else:
            assert False, f'Invalid cov_off_diagonal: {cov_off_diagonal}'

        self.weigh = torch.ones((1, self.K), requires_grad=False).to(self.args.device)
        if average == 'weighted':
            self.weigh = nn.Parameter(self.weigh, requires_grad=True)
        else:
            assert average == 'fixed', f"Invalid average: {average}"

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.diag
        if True:
            logvar = logvar.tanh()  # .exp()
        y = y * logvar
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(logvar)), dim=2) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def update_parameters(self, kernel_dict):
        tri = []
        means = []
        weigh = []
        diag = []
        for key, value in kernel_dict.items():  # detach and clone
            tri.append(copy.deepcopy(value.tri.detach().clone()))
            means.append(copy.deepcopy(value.means.detach().clone()))
            weigh.append(copy.deepcopy(value.weigh.detach().clone()))
            diag.append(copy.deepcopy(value.diag.detach().clone()))

        self.tri = nn.Parameter(torch.cat(tri, dim=1), requires_grad=True).to(self.args.device)
        self.means = nn.Parameter(torch.cat(means, dim=0), requires_grad=True).to(self.args.device)
        self.weigh = nn.Parameter(torch.cat(weigh, dim=-1), requires_grad=True).to(self.args.device)
        self.diag = nn.Parameter(torch.cat(diag, dim=1), requires_grad=True).to(self.args.device)

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def forward(self, x):
        y = -self.logpdf(x)
        return torch.mean(y)


class MIClassificationStyleEmdedding(torch.nn.Module):
    """
    AAAI 2019 model 2
    """

    def __init__(self, args, parameter_dataloader):
        super(MIClassificationStyleEmdedding, self).__init__()
        self.args = args
        self.parameter_dataloader = parameter_dataloader
        self.encoder = EncoderRNN(args, args.number_of_tokens, args.hidden_dim, args.number_of_layers)
        self.loss_classif = torch.nn.NLLLoss()

        self.proj_content = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.LeakyReLU(),
                                          nn.Linear(args.hidden_dim, args.content_dim))

        # D_\gamma
        self.d_gamma = ClassifierGamma(args.content_dim + args.number_of_styles,
                                       args.number_of_styles)

        self.number_of_point_kernel = args.kernel_size

        self.mi_estimator = eval(args.mi_estimator_name)(args.content_dim, 1, args.content_dim)

        self.advers_classifier = Classifier(args.content_dim, args.number_of_styles, args.use_complex_classifier)
        self.downstream_classifier = Classifier(args.content_dim, args.number_of_styles,
                                                # args.number_of_downstream_labels
                                                args.use_complex_classifier)

        # Loss : paper multipliers
        self.mul_mi = self.args.mul_mi

    def forward(self, input_tensor, protected_labels, downstream_labels):
        count = 0
        for step, batch in enumerate(self.parameter_dataloader):
            if count == self.args.reny_training + 1:
                break
            count += 1
            inputs_reny = batch['line'].to(self.args.device)
            labels_reny_downstream = batch['downstream_labels'].to(self.args.device)
            labels_reny_protected = batch['label'].to(self.args.device)

            ###################################
            ##### Compute h(Z|Y=y) & h(Z) #####
            ###################################
            frozen_params(self.downstream_classifier)
            frozen_params(self.proj_content)
            free_params(self.mi_estimator)
            frozen_params(self.encoder)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny.clone(), encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)

            training_loss_mi = self.mi_estimator.learning_loss(torch.sum(content, 0),
                                                               labels_reny_protected.unsqueeze(-1))

            if self.training:
                training_loss_mi.backward()
                torch.nn.utils.clip_grad_norm_(self.mi_estimator.parameters(),
                                               self.args.max_grad_norm)
                self.args.optimizer.step()
                self.mi_estimator.zero_grad()

            ##########################
            # Update downstream taks #
            ##########################
            free_params(self.downstream_classifier)
            frozen_params(self.proj_content)
            frozen_params(self.mi_estimator)
            frozen_params(self.encoder)

            inputs_reny_gen = inputs_reny.clone()
            if self.args.add_noise:
                inputs_reny_gen = corrupt_input(self, inputs_reny_gen)

            encoder_hidden = self.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(inputs_reny_gen, encoder_hidden)

            # Content space
            content = self.proj_content(encoder_hidden)
            down_pred = self.downstream_classifier(content)
            loss_down_classifier = self.loss_classif(down_pred, labels_reny_downstream)

            if self.training:
                loss_down_classifier.backward()
                torch.nn.utils.clip_grad_norm_(self.downstream_classifier.parameters(), self.args.max_grad_norm)
                self.args.optimizer.step()
                self.downstream_classifier.zero_grad()

        ###############################
        # Update Genloss + \lambda * MI
        ###############################
        input_tensor_gen = input_tensor.clone()
        if self.args.add_noise:
            input_tensor_gen = corrupt_input(self, input_tensor_gen)
        input_tensor_mi = input_tensor_gen.clone()
        if self.training:
            frozen_params(self.downstream_classifier)
            frozen_params(self.mi_estimator)
            free_params(self.proj_content)
            free_params(self.encoder)

        # Compute MI with protected_labels
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_mi, encoder_hidden)
        content = self.proj_content(encoder_hidden)

        loss_mi = self.mi_estimator(torch.sum(content, 0), protected_labels.unsqueeze(-1))

        ###################
        # Downstream loss #
        ###################
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor_gen, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_pred = self.downstream_classifier(content)
        loss_down_classifier = self.loss_classif(down_pred, downstream_labels)

        # Compute All Losses for MI
        gradient_encoder, gradient_content_proj = torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(
            self.args.device)
        if self.training:
            loss = loss_down_classifier + self.args.mul_mi * loss_mi
            loss.backward()
            gradient_encoder, gradient_content_proj = comput_gradient_norm(
                self.encoder), comput_gradient_norm(self.proj_content)
            torch.nn.utils.clip_grad_norm_(self.proj_content.parameters(), self.args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.max_grad_norm)
            self.args.optimizer.step()
            self.args.scheduler.step()
            self.proj_content.zero_grad()
            self.encoder.zero_grad()

        losses_dic = {'loss_down_classifier': loss_down_classifier,
                      'loss_mi': self.args.mul_mi * loss_mi,
                      'gradient_encoder': gradient_encoder,
                      'gradient_content_proj': gradient_content_proj}
        return losses_dic,

    def predict_latent_space(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Ensure Style not in content space
        content = self.proj_content(encoder_hidden)
        return content

    def predict_downstream(self, inputs):
        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)

        # Content space
        content = self.proj_content(encoder_hidden)
        down_preds = self.downstream_classifier(content)

        return down_preds
