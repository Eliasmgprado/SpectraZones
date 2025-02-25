from torch import nn


class AutoencoderNN(nn.Module):
    """
    Deep AutoEncoder NN

    """

    def __init__(self, input_size, start_filter=256, depth=1):
        """
        Parameters
        ----------
        input_size : int
            Input data feature size.
        start_filter : int
            Start filter size.
        depth : int
            Number of layers.
        """
        super(AutoencoderNN, self).__init__()
        depth = int(depth)

        self.encoder = nn.Sequential()
        self.encoder.add_module("lin_1", nn.Linear(input_size, int(start_filter)))
        self.encoder.add_module("relu_1", nn.ReLU(True))

        for n in range(1, depth):
            self.encoder.add_module(
                f"lin_{n+1}",
                nn.Linear(
                    int(start_filter / (2 ** (n - 1))), int(start_filter / (2**n))
                ),
            )
            self.encoder.add_module(f"relu_{n+1}", nn.ReLU(True))

        self.encoder.add_module(
            f"lin_{depth+1}",
            nn.Linear(
                int(start_filter / (2 ** (depth - 1))), int(start_filter / (2**depth))
            ),
        )

        self.decoder = nn.Sequential()
        self.decoder.add_module(
            "lin_1",
            nn.Linear(
                int(start_filter / (2**depth)), int(start_filter / (2 ** (depth - 1)))
            ),
        )
        self.decoder.add_module("relu_1", nn.ReLU(True))

        for n in range(depth - 1, 0, -1):
            self.decoder.add_module(
                f"lin_{depth-n+1}",
                nn.Linear(
                    int(start_filter / (2**n)), int(start_filter / (2 ** (n - 1)))
                ),
            )
            self.decoder.add_module(f"relu_{depth-n+1}", nn.ReLU(True))

        self.decoder.add_module(
            f"lin_{depth+1}", nn.Linear(int(start_filter), input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
