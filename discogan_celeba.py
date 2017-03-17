# coding: UTF-8
import argparse
import os
import csv
import random

import numpy as np
from PIL import Image
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, root, size=64, random=True):
        self.paths = paths
        self.root = root
        self.size = size
        self.random = random

    def __len__(self):
        return len(self.paths)

    def read_image_as_array(self, path):
        f = Image.open(path)
        f = f.resize((109, 89), Image.ANTIALIAS)
        try:
            image = np.asarray(f, dtype=np.float32)
        finally:
            if hasattr(f, 'close'):
                f.close()
        return image.transpose((2, 0, 1))

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [-1, 1] value

        path = os.path.join(self.root, self.paths[i])
        image = self.read_image_as_array(path)
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - self.size)
            left = random.randint(0, w - self.size)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - self.size) // 2
            left = (w - self.size) // 2
        bottom = top + self.size
        right = left + self.size

        image = image[:, top:bottom, left:right]
        image *= (2 / 255)
        image -= 1
        return image


class Generator(chainer.Chain):
    def __init__(self, ):
        super(Generator, self).__init__(
            conv1=L.Convolution2D(None, 128, 4, 2, 1),
            conv2=L.Convolution2D(None, 128, 4, 2, 1),
            norm2=L.BatchNormalization(128),
            conv3=L.Convolution2D(None, 64, 4, 2, 1),
            norm3=L.BatchNormalization(64),
            conv4=L.Convolution2D(None, 32, 4, 2, 1),
            norm4=L.BatchNormalization(32),

            deconv1=L.Deconvolution2D(None, 64, 4, 2, 1),
            dnorm1=L.BatchNormalization(64),
            deconv2=L.Deconvolution2D(None, 128, 4, 2, 1),
            dnorm2=L.BatchNormalization(128),
            deconv3=L.Deconvolution2D(None, 128, 4, 2, 1),
            dnorm3=L.BatchNormalization(128),
            deconv4=L.Deconvolution2D(None, 3, 4, 2, 1),
            )

    def __call__(self, x, test=False):
        # convolution
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1), test=test))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2), test=test))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3), test=test))

        # deconvolution
        dh1 = F.leaky_relu(self.dnorm1(self.deconv1(h4), test=test))
        dh2 = F.leaky_relu(self.dnorm2(self.deconv2(dh1), test=test))
        dh3 = F.leaky_relu(self.dnorm3(self.deconv3(dh2), test=test))
        y = F.tanh(self.deconv4(dh3))
        return y


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            conv1=L.Convolution2D(None, 128, 4, 2, 1),
            conv2=L.Convolution2D(None, 128, 4, 2, 1),
            norm2=L.BatchNormalization(128),
            conv3=L.Convolution2D(None, 64, 4, 2, 1),
            norm3=L.BatchNormalization(64),
            conv4=L.Convolution2D(None, 32, 4, 2, 1),
            norm4=L.BatchNormalization(32),
            fc=L.Linear(None, 1)
            )

    def __call__(self, x, test=False):
        # convolution
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1), test=test))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2), test=test))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3), test=test))

        # full connect
        y = self.fc(h4)
        return y


class DiscoGANUpdater(training.StandardUpdater):
    def __init__(self, iterator_a, iterator_b, opt_g_ab, opt_g_ba,
                 opt_d_a, opt_d_b, device):
        self._iterators = {'main': iterator_a, 'second': iterator_b}
        self.generator_ab = opt_g_ab.target
        self.generator_ba = opt_g_ba.target
        self.discriminator_a = opt_d_a.target
        self.discriminator_b = opt_d_b.target
        self._optimizers = {'generator_ab': opt_g_ab,
                            'generator_ba': opt_g_ba,
                            'discriminator_a': opt_d_a,
                            'discriminator_b': opt_d_b}
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        # read data
        batch_a = self._iterators['main'].next()
        x_a = self.converter(batch_a, self.device)

        batch_b = self._iterators['second'].next()
        x_b = self.converter(batch_b, self.device)

        batchsize = x_a.shape[0]

        # conversion
        x_ab = self.generator_ab(x_a)
        x_ba = self.generator_ba(x_b)

        # reconversion
        x_aba = self.generator_ba(x_ab)
        x_bab = self.generator_ab(x_ba)

        # discriminate
        y_a = self.discriminator_a(F.concat((x_a, x_ba), 0))
        y_a_real, y_a_fake = F.split_axis(y_a, 2, 0)

        y_b = self.discriminator_b(F.concat((x_b, x_ab), 0))
        y_b_real, y_b_fake = F.split_axis(y_b, 2, 0)

        # compute loss
        # SCE(x, 0) = softplus(x)
        # SCE(x, 1) = softplus(-x)
        loss_gan_real = F.sum(
            F.softplus(-y_a_real) + F.softplus(-y_b_real)) / batchsize
        loss_gan_fake = F.sum(
            F.softplus(y_a_fake) + F.softplus(y_b_fake)) / batchsize

        loss_const_a = F.mean_squared_error(x_a, x_aba)
        loss_const_b = F.mean_squared_error(x_b, x_bab)

        loss_gen = - loss_gan_fake + loss_const_a + loss_const_b
        loss_dis = loss_gan_real + loss_gan_fake

        # update
        self.generator_ab.cleargrads()
        self.generator_ba.cleargrads()
        loss_gen.backward()
        self._optimizers['generator_ab'].update()
        self._optimizers['generator_ba'].update()

        self.discriminator_a.cleargrads()
        self.discriminator_b.cleargrads()
        loss_dis.backward()
        self._optimizers['discriminator_a'].update()
        self._optimizers['discriminator_b'].update()

        # report
        chainer.reporter.report({
            'loss/gan/real': loss_gan_real,
            'loss/gan/fake': loss_gan_fake,
            'loss/const': loss_const_a + loss_const_b})


def main():
    parser = argparse.ArgumentParser(description='DiscoGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=200,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', type=int, default=2,
                        help='Number of parallel data loading processes')
    parser.add_argument('--directory', '-d', default='./',
                        help='root directory of CelebA Dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    generator_ab = Generator()
    generator_ba = Generator()
    discriminator_a = Discriminator()
    discriminator_b = Discriminator()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator_ab.to_gpu()
        generator_ba.to_gpu()
        discriminator_a.to_gpu()
        discriminator_b.to_gpu()

    opt_g_ab = chainer.optimizers.Adam(2e-4, beta1=0.5, beta2=0.999)
    opt_g_ab.setup(generator_ab)
    opt_g_ab.add_hook(chainer.optimizer.WeightDecay(1e-4))
    opt_g_ba = chainer.optimizers.Adam(2e-4, beta1=0.5, beta2=0.999)
    opt_g_ba.setup(generator_ba)
    opt_g_ba.add_hook(chainer.optimizer.WeightDecay(1e-4))

    opt_d_a = chainer.optimizers.Adam(2e-4, beta1=0.5, beta2=0.999)
    opt_d_a.setup(discriminator_a)
    opt_d_a.add_hook(chainer.optimizer.WeightDecay(1e-4))
    opt_d_b = chainer.optimizers.Adam(2e-4, beta1=0.5, beta2=0.999)
    opt_d_b.setup(discriminator_b)
    opt_d_b.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # load data
    list_a = []
    list_b = []
    with open(os.path.join(
            args.directory, 'Anno/list_attr_celeba.txt'), 'r') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        number = next(reader)
        header = next(reader)
        for row in reader:
            path = row[0].replace('jpg', 'png')
            # print(row[0], path)
            if row[21] == '1':
                list_a.append(path)
            elif row[21] == '-1':
                list_b.append(path)

    train_a = PreprocessedDataset(list_a[:80000], os.path.join(
        args.directory, 'Img/img_align_celeba_png/'))
    train_b = PreprocessedDataset(list_b[:80000], os.path.join(
        args.directory, 'Img/img_align_celeba_png/'))
    valid_a = PreprocessedDataset(list_a[80000:80010], os.path.join(
        args.directory, 'Img/img_align_celeba_png/'), random=False)
    valid_b = PreprocessedDataset(list_b[80000:80010], os.path.join(
        args.directory, 'Img/img_align_celeba_png/'), random=False)

    train_iter_a = chainer.iterators.MultiprocessIterator(
        train_a, args.batchsize, n_processes=args.loaderjob // 2)
    train_iter_b = chainer.iterators.MultiprocessIterator(
        train_b, args.batchsize, n_processes=args.loaderjob // 2)
    valid_iter_a = chainer.iterators.SerialIterator(valid_a, 10)
    valid_iter_b = chainer.iterators.SerialIterator(valid_b, 10)

    updater = DiscoGANUpdater(train_iter_a, train_iter_b, opt_g_ab, opt_g_ba,
                              opt_d_a, opt_d_b, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    def out_generated_image(iterator_a, iterator_b,
                            generator_ab, generator_ba, device, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            # read data
            batch_a = iterator_a.next()
            x_a = convert.concat_examples(batch_a, device)

            batch_b = iterator_b.next()
            x_b = convert.concat_examples(batch_b, device)

            # conversion
            x_ab = generator_ab(x_a, test=True)
            x_ba = generator_ba(x_b, test=True)

            # to cpu
            x_a = chainer.cuda.to_cpu(x_a)
            x_b = chainer.cuda.to_cpu(x_b)
            x_ab = chainer.cuda.to_cpu(x_ab.data)
            x_ba = chainer.cuda.to_cpu(x_ba.data)

            # reshape
            x_a = np.concatenate((x_a, x_ab), 0)
            x_a = x_a.reshape(2, 10, 3, 64, 64)
            x_a = x_a.transpose(0, 3, 1, 4, 2)
            x_a = x_a.reshape((2 * 64, 10 * 64, 3))

            x_b = np.concatenate((x_b, x_ba), 0)
            x_b = x_b.reshape(2, 10, 3, 64, 64)
            x_b = x_b.transpose(0, 3, 1, 4, 2)
            x_b = x_b.reshape((2 * 64, 10 * 64, 3))

            # to [0, 255]
            x_a += 1
            x_a *= (255 / 2)
            x_a = np.asarray(np.clip(x_a, 0, 255), dtype=np.uint8)

            x_b += 1
            x_b *= (255 / 2)
            x_b = np.asarray(np.clip(x_b, 0, 255), dtype=np.uint8)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>5}.png'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x_a).save(preview_path.replace('.png', '_a.png'))
            Image.fromarray(x_b).save(preview_path.replace('.png', '_b.png'))
        return make_image

    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(['loss/const'],
                              'epoch', file_name='const.png'))
    trainer.extend(
        extensions.PlotReport(
            ['loss/gan/real', 'loss/gan/fake'], 'epoch', file_name='gan.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'loss/gan/real', 'loss/gan/fake',
         'loss/const', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_generated_image(valid_iter_a, valid_iter_b,
                                       generator_ab, generator_ba,
                                       args.gpu, args.out),
                   trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
