import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from mi_estimators import *
from data_utils import *


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.args = config.args
        device = config.args.device
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.deterministic = config.deterministic
        self.ib_dim = config.ib_dim
        self.ib = config.ib
        self.activation = config.activation
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if self.ib or self.deterministic:
            self.kl_annealing = config.kl_annealing
            self.hidden_dim = config.hidden_dim
            intermediate_dim = (self.hidden_dim + config.hidden_size) // 2
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation])
            self.beta = config.beta
            self.sample_size = config.sample_size
            self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim)
            self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))
            self.classifier = nn.Linear(self.ib_dim, self.config.num_labels)
            if self.args.use_mi_estimation:
                if self.args.name_mi_estimator == 'KERNEL_E':
                    self.mi_estimator = MultiGaussKernelEE(device, config.args.number_of_samples, self.ib_dim).to(
                        device)
                elif self.args.name_mi_estimator == 'KERNEL_A':
                    self.mi_estimator = MIKernelEstimator(device, config.args.number_of_samples, self.ib_dim,
                                                          config.hidden_size,
                                                          use_joint=False).to(device)
                elif self.args.name_mi_estimator in ['CLUBSample', 'MINE', 'L1OutUB', 'NWJ', 'InfoNCE']:
                    self.mi_estimator = eval(self.args.name_mi_estimator)(config.hidden_size, self.ib_dim,
                                                                     intermediate_dim).to(device)
                elif self.args.name_mi_estimator in ["DOE"]:
                    self.mi_estimator = eval(self.args.name_mi_estimator)(self.ib_dim, config.hidden_size).to(
                        device)
                else:
                    raise NotImplementedError
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).to(device)
        return mu + std * z

    def get_logits(self, z, mu, sampling_type):
        if sampling_type == "iid":
            logits = self.classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits

    def sampled_loss(self, logits, mean_logits, labels, sampling_type):
        if sampling_type == "iid":
            # During the training, computes the loss with the sampled embeddings.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(mean_logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
        return loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            sampling_type="iid",
            epoch=1,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """

        final_outputs = {}
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        loss = {}

        if self.deterministic:
            pooled_output = self.mlp(pooled_output)
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            final_outputs["z"] = mu
            sampled_logits, logits = self.get_logits(mu, mu, sampling_type='argmax')  # always deterministic
            if labels is not None:
                loss["loss"] = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type='argmax')

        elif self.ib:
            # sample X sample for the dataset
            count = 0
            if self.training and self.args.use_mi_estimation:
                free_params(self.mi_estimator)
                frozen_params(self.bert)
                frozen_params(self.mlp)
                frozen_params(self.emb2mu)
                frozen_params(self.emb2std)
                frozen_params(self.classifier)
                for batch_estimator in tqdm(self.data_loader, desc="Estimator Training"):
                    count += 1
                    if count > self.args.loader_number:
                        break
                    batch_estimator = tuple(t.to(self.args.device) for t in batch_estimator)
                    inputs_estimator = {"input_ids": batch_estimator[0], "attention_mask": batch_estimator[1],
                                        "labels": batch_estimator[3]}
                    outputs_estimator = self.bert(
                        batch_estimator[0],
                        attention_mask=batch_estimator[1],
                        token_type_ids=None,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                    )
                    pooled_output_estimator = outputs_estimator[1]
                    pooled_output_estimator = self.dropout(pooled_output_estimator)
                    pooled_output_mlp_estimator = self.mlp(pooled_output_estimator)
                    batch_size_estimator = pooled_output_mlp_estimator.shape[0]
                    mu_estimator, std_estimator = self.estimate(pooled_output_mlp_estimator, self.emb2mu, self.emb2std)
                    z_estimator = self.reparameterize(mu_estimator, std_estimator)
                    if self.args.name_mi_estimator == 'KERNEL_E':
                        training_loss_estimator = self.mi_estimator.learning_loss(z_estimator[0, :, :],
                                                                                  None)  # mi estimation
                    elif self.args.name_mi_estimator == 'KERNEL_A':
                        training_loss_estimator = self.mi_estimator.learning_loss(z_estimator[0, :, :],
                                                                                  pooled_output_estimator)  # mi estimation
                    else:
                        training_loss_estimator = self.mi_estimator.learning_loss(pooled_output_estimator,
                                                                                  z_estimator[0, :, :])  # mi estimation
                    training_loss_estimator.backward()
                    self.optimizer.step()
                    self.mi_estimator.zero_grad()
            if self.args.use_mi_estimation:
                frozen_params(self.mi_estimator)
                free_params(self.bert)
                free_params(self.mlp)
                free_params(self.emb2mu)
                free_params(self.emb2std)
                free_params(self.classifier)

            pooled_output_mlp = self.mlp(pooled_output)
            batch_size = pooled_output_mlp.shape[0]
            mu, std = self.estimate(pooled_output_mlp, self.emb2mu, self.emb2std)
            mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))

            z = self.reparameterize(mu, std)
            if self.args.use_mi_estimation:
                if self.args.name_mi_estimator == 'KERNEL_E':
                    mi_loss = self.mi_estimator(z[0, :, :], None)  # mi estimation
                elif self.args.name_mi_estimator == 'KERNEL_A':
                    mi_loss = self.mi_estimator(z[0, :, :], pooled_output)  # mi estimation  # mi estimation
                else:
                    mi_loss = self.mi_estimator(pooled_output, z[0, :, :])  # mi estimation
                if self.args.name_mi_estimator == 'KERNEL_E':
                    mi_loss = torch.abs(
                        mi_loss - 1 / 2 * torch.mean(torch.sum(2 * torch.log(torch.clamp(std, min=1e-8)), dim=1)))
            else:
                mi_loss = self.kl_div(mu, std, mu_p, std_p)
            final_outputs["z"] = mu

            if self.kl_annealing == "linear":
                beta = min(1.0, epoch * self.beta)

            sampled_logits, logits = self.get_logits(z, mu, sampling_type)
            if labels is not None:
                ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
                loss["loss"] = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * mi_loss
        else:
            final_outputs["z"] = pooled_output
            logits = self.classifier(pooled_output)
            mi_loss = torch.tensor(0.0).to(self.args.device)
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss["loss"] = loss_fct(logits.view(-1), labels.float().view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss["loss"] = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                ce_loss = loss["loss"]
        try:
            print(ce_loss)
        except:
            ce_loss = torch.tensor(0.0).to(self.args.device)
        final_outputs.update(
            {"logits": logits, "loss": loss, "hidden_attention": outputs[2:], 'mi_loss': mi_loss, 'ce_loss': ce_loss})
        return final_outputs
