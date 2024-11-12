import os

from PIL import Image

from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        name = "boreas"
        dataroot = os.path.expanduser("~/data/Weather/boreas_unpaired")
        weathers = ["rainy", "snowy"]

        # Maximum training image size in RTX 3090 24GB
        # issue: https://github.com/taesungp/contrastive-unpaired-translation/issues/139#issuecomment-1165577005
        max_crop_size = 768
        max_load_size = round(max_crop_size * 1.1)
        resampling = Image.Resampling.LANCZOS.name

        return [
            Options(
                dataroot=dataroot,
                dataset_mode="unaligned_weather",
                data_domainA="clear",
                data_domainB=weather,
                name=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                CUT_mode="CUT",
                display_env=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                max_load_size=max_load_size,
                max_crop_size=max_crop_size,
                resampling=resampling,
            )
            for weather in weathers
        ]

    def commands(self):
        lr = 0.0002
        lambda_GAN = 1.0
        n_epochs = 100
        n_epochs_decay = 100

        return [
            "python train.py "
            + str(
                opt.set(
                    name=f"{opt.kvs['name']}_epochs_{n_epochs + n_epochs_decay}_no_seed",
                    display_env=f"{opt.kvs['name']}_epochs_{n_epochs + n_epochs_decay}_no_seed",
                    data_domainA=os.path.join("train", opt.kvs["data_domainA"]),
                    data_domainB=os.path.join("train", opt.kvs["data_domainB"]),
                    lr=lr,
                    lambda_GAN=lambda_GAN,
                    n_epochs=n_epochs,
                    n_epochs_decay=n_epochs_decay,
                )
            )
            for opt in self.common_options()
        ]

    def test_commands(self):
        return [
            "python test.py "
            + str(
                opt.remove("display_env").set(
                    name=opt.kvs["name"],
                    display_env=opt.kvs["display_env"],
                    data_domainA=os.path.join("val", opt.kvs["data_domainA"]),
                    data_domainB=os.path.join("val", opt.kvs["data_domainB"]),
                    num_test=1000,
                )
            )
            for opt in self.common_options()
        ]
