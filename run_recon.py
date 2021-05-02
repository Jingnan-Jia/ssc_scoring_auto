
from torchsummary import summary


class ReconNet(nn.Module):
    def __init__(reg_net, input_size=512):
        self.reg_net = reg_net
        self.features = reg_net.features  # encoder
        self.decoder = self._build_dec_from_enc()

    def _build_dec_from_enc(self):
        decoder_ls = []
        for layer in self.features[::-1]:
            if isinstance(layer, torch.nn.Conv2d):
                in, out, kernel_size =
                decoder_ls.extend = [nn.Conv2d(out, in,
                                     kernel_size = kernel_size, padding = 1),  # in and out are reflect
                nn.BatchNorm2d( in),
                nn.ReLU(inplace=True)]
                elif isinstance(layer, torch.nn.MaxPool2d):
                decoder_ls.extend[nn.ConvTranspose2d(out, in, kernel_size = 2, stride = 2)]
                else:
                pass
        while (isinstance(decoder_ls[-1], nn.ReLU)) or (isinstance(decoder_ls[-1], nn.BatchNorm2d)):
            decoder_ls.pop()

    def forward(self, x):
        latent = self.features(x)
        out = self.decoder(latent)

        return out


reg_net = get_net('vgg11bn')
recon_net = ReconNet(reg_net)


class ReconDatasetd(Dataset):
    def __init__(self, data_x_names, transform=None):
        self.data_x_names = data_x_names
        self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in self.data_x_names]
        self.data_x_np = [i[0] for i in self.data_x]

    def __len__(self):

    def __getitem__(self, idx):
        img = self.data_x_np[idx]
        slice_nb = random.randint(0, len(img))
        slice = img[slice_nb]
        if self.transform:
            image = self.transform(slice)
        data = {'image_key': slice, 'label_key': slice}
        return data


def get_transform_recon():
    transform = [RandomHorizontalFlipd(), RandomVerticalFlipd()]
    transform = transforms.Compose(transform)
    return transform
