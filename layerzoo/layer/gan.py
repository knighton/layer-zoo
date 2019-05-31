import torch
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.utils import save_image

from .base import Layer, summarize


class GAN(Layer):
    """
    Converts the input to dense embeddings via a modified GAN setup.

    Training steps:

        1. real images
             [judge]
           is genuine (yes)

        2. fake embeddings
             [render]
           fake images
             [judge]
           is genuine (no)

        3. real images
             [embed]
           embeddings
             [render]
           images
             [judge]
           is genuine (yes)
    """

    def __init__(self, observer, artist, rater):
        super().__init__()
        self.observers = observer,
        self.artists = artist,
        self.raters = rater,
        lr = 5e-3
        self.observer_optimizers = Adam(observer.parameters(), lr=lr),
        self.artist_optimizers = Adam(artist.parameters(), lr=lr),
        self.rater_optimizers = Adam(rater.parameters(), lr=lr * 0.5),
        self.batch_id = 0

    # TODO: the rest of this device stuff.
    def cpu(self):
        observer, = self.observers
        artist, = self.artists
        rater, = self.raters
        observer.cpu()
        artist.cpu()
        rater.cpu()
        super().cpu()

    def cuda(self):
        observer, = self.observers
        artist, = self.artists
        rater, = self.raters
        observer.cuda()
        artist.cuda()
        rater.cuda()
        super().cuda()

    def train_on_batch(self, x):
        batch_size = x.shape[0]

        found_valid = torch.ones(batch_size, 1)
        found_fake = torch.zeros(batch_size, 1)
        if x.is_cuda:
            found_valid = found_valid.cuda()
            found_fake = found_fake.cuda()

        observer, = self.observers
        artist, = self.artists
        rater, = self.raters

        observer_optimizer, = self.observer_optimizers
        artist_optimizer, = self.artist_optimizers
        rater_optimizer, = self.rater_optimizers

        bce = F.binary_cross_entropy

        # Train the observer.
        #
        # Successful observers enable their artists to be successful.
        observer_optimizer.zero_grad()
        observations = observer(x)
        recreated_images = artist(observations)
        observer_loss = bce(rater(recreated_images), found_valid)
        observer_loss.backward()
        observer_optimizer.step()

        # Train the artist.
        #
        # Successful artists fool their raters.
        artist_optimizer.zero_grad()
        fake_observations = torch.randn(*observations.shape).tanh()
        if x.is_cuda:
            fake_observations = fake_observations.cuda()
        fake_x = artist(fake_observations)
        artist_loss = bce(rater(fake_x), found_valid)
        artist_loss.backward()
        artist_optimizer.step()

        # Train the rater.
        #
        # Successful raters are not fooled.
        rater_optimizer.zero_grad()
        real_loss = bce(rater(x), found_valid)
        fake_loss = bce(rater(fake_x.detach()), found_fake)
        rater_loss = (real_loss + fake_loss) / 2
        rater_loss.backward()
        rater_optimizer.step()

        line = 'batch %d observer %.3f artist %.3f rater %.3f' % \
            (self.batch_id, observer_loss.item(), artist_loss.item(),
             rater_loss.item())
        print(line)

        if not self.batch_id % 50:
            import os
            os.makedirs('images/', exist_ok=True)
            save_image(fake_x.data, "images/%07d.png" % self.batch_id,
                       nrow=4, normalize=True)

        self.batch_id += 1

    def forward_inner(self, x):
        if self.training:
            self.train_on_batch(x)
        observer, = self.observers
        return observer(x)
