import os

from PIL import Image

from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        name = "nuscenes_foggy_zurich"
        dataroot = os.path.expanduser("~/data/Weather/nuScenes_Foggy_Zurich_unpaired")
        weathers = ["rainy", "foggy"]

        # Maximum training image size in RTX 3090 24GB: 768
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
                name=f"{name}_santa_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                display_env=f"{name}_santa_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                max_load_size=max_load_size,
                max_crop_size=max_crop_size,
                resampling=resampling,
            )
            for weather in weathers
        ]

    def commands(self):
        lr = 0.0002
        lambda_path = 0.1
        n_epochs = 200
        n_epochs_decay = 200
        batch_size = 1
        tag = "santa"

        # parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--lambda_idt', type=float, default=5.0, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--lambda_path', type=float, default=0.01, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--path_layers', type=str, default='0,3,6,10,14', help='compute NCE loss on which layers')
        # parser.add_argument('--style_dim', type=int, default=8, help='weight for NCE loss: NCE(G(X), X)')
        # parser.add_argument('--path_interval_min', type=float, default=0.05, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--path_interval_max', type=float, default=0.10, help='weight for GAN loss：GAN(G(X))')
        # parser.add_argument('--noise_std', type=float, default=1.0, help='compute NCE loss on which layers')
        # parser.add_argument('--tag', type=str, default='debug', help='compute NCE loss on which layers')

        return [
            "python train.py "
            + str(
                opt.set(
                    name=f"{opt.kvs['name']}_epochs_{n_epochs + n_epochs_decay}_bs_{batch_size}_no_seed",
                    display_id=1,
                    display_env=f"{opt.kvs['name']}_epochs_{n_epochs + n_epochs_decay}_bs_{batch_size}_no_seed",
                    # data_domainA=os.path.join("train", opt.kvs["data_domainA"]),
                    # data_domainB=os.path.join("train", opt.kvs["data_domainB"]),
                    lr=lr,
                    lambda_path=lambda_path,
                    n_epochs=n_epochs,
                    n_epochs_decay=n_epochs_decay,
                    batch_size=batch_size,
                    tag=tag,
                )
            )
            for opt in self.common_options()
        ]

    def test_commands(self):
        return [
            "python test.py "
            + str(
                opt.remove("display_env").set(
                    # name="nuscenes_foggy_zurich_CUT_foggy_load_563_crop_512_epochs_100_bs_2_no_seed",
                    # data_domainA=os.path.join("val", opt.kvs["data_domainA"]),
                    # data_domainB=os.path.join("val", opt.kvs["data_domainB"]),
                    num_test=1000,
                    eval=True,
                )
            )
            for opt in self.common_options()
        ]
