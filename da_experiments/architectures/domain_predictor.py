import torch.nn as nn


class DomainPredictor(nn.Module):
    def __init__(self, args, input_dim, num_domains):
        super(DomainPredictor, self).__init__()
        self.latent_dim = args.latent_dim_d
        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(nn.Linear(input_dim, self.latent_dim),
                                   self.activation,
                                   nn.Linear(self.latent_dim, num_domains))

    def forward(self, x):
        x = self.block(x)
        return x


    # def label_predictor(self):
    #     # with tf.variable_scope('label_predictor_fc1'):
    #     #     fc_1 = layers.fully_connected(inputs=self.features_for_prediction, num_outputs=self.latent_dim,
    #     #                                     activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #     with tf.variable_scope('label_predictor_logits'):
    #         logits = layers.fully_connected(inputs=self.features_c_for_prediction, num_outputs=self.num_labels,
    #                                         activation_fn=None, weights_initializer=self.initializer)

    #     self.y_pred = tf.nn.softmax(logits)
    #     self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y))
    #     self.y_acc = utils.predictor_accuracy(self.y_pred, self.y)
    # def domain_predictor(self, reuse = False):
    #     with tf.variable_scope('domain_predictor_fc1', reuse = reuse):
    #         fc_1 = layers.fully_connected(inputs=self.features_d, num_outputs=self.latent_dim,
    #                                       activation_fn=tf.nn.relu, weights_initializer=self.initializer)
    #     with tf.variable_scope('domain_predictor_logits', reuse = reuse):
    #         self.d_logits = layers.fully_connected(inputs=fc_1, num_outputs=self.num_domains,
    #                                           activation_fn=None, weights_initializer=self.initializer)


    #     logits_real = tf.slice(self.d_logits, [0, 0], [self.batch_size, -1])
    #     logits_fake = tf.slice(self.d_logits, [self.batch_size, 0], [self.batch_size * self.num_targets, -1])

    #     label_real = tf.slice(self.domains, [0, 0], [self.batch_size, -1])
    #     label_fake = tf.slice(self.domains, [self.batch_size, 0], [self.batch_size * self.num_targets, -1])
    #     label_pseudo = tf.ones(label_fake.shape) - label_fake

    #     self.d_pred = tf.nn.sigmoid(self.d_logits)
    #     real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = label_real))
    #     fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = label_fake))
    #     self.d_loss = real_d_loss + self.reg_tgt * fake_d_loss
    #     self.d_acc = utils.predictor_accuracy(self.d_pred,self.domains)