import torch
import torch.nn as nn

class U3Net(nn.Module):
    def __init__(self):
        super(U3Net, self).__init__()

        def conv_block(input_channels, output_channels):
            return nn.Sequential(
                nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(output_channels),
                nn.ReLU(inplace=True)
            )

        self.max_pool = nn.MaxPool3d(2)

        self.encoder_block1 = nn.Sequential(conv_block(1, 32), conv_block(32, 64))
        self.encoder_block2 = nn.Sequential(conv_block(64, 64), conv_block(64, 128))
        self.encoder_block3 = nn.Sequential(conv_block(128, 128), conv_block(128, 256))
        self.encoder_block4 = nn.Sequential(conv_block(256, 256), conv_block(256, 512))

        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2),
            conv_block(768, 256),
            conv_block(256, 256)
        )

        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            conv_block(384, 128),
            conv_block(128, 128)
        )

        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            conv_block(192, 64),
            conv_block(64, 64)
        )

        self.output_layer = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder_block1(x)
        pooled1 = self.max_pool(enc1)

        enc2 = self.encoder_block2(pooled1)
        pooled2 = self.max_pool(enc2)

        enc3 = self.encoder_block3(pooled2)
        pooled3 = self.max_pool(enc3)

        enc4 = self.encoder_block4(pooled3)

        dec1_input = torch.cat([enc3, self.decoder_block1[0](enc4)], dim=1)
        dec1 = self.decoder_block1[1:](dec1_input)

        dec2_input = torch.cat([enc2, self.decoder_block2[0](dec1)], dim=1)
        dec2 = self.decoder_block2[1:](dec2_input)

        dec3_input = torch.cat([enc1, self.decoder_block3[0](dec2)], dim=1)
        dec3 = self.decoder_block3[1:](dec3_input)

        return self.output_layer(dec3)
