from lightning_model import NuWave2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
from tqdm import tqdm
from scipy.io.wavfile import write as swrite

from utils.stft import STFTMag

import csv
import time


def save_results_to_csv(results, filename="results.csv"):
    # Check if file exists. If not, create it and write headers
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Sample Rate",
                    "SNR (mean)",
                    "SNR (std)",
                    "Base SNR (mean)",
                    "Base SNR (std)",
                    "LSD (mean)",
                    "LSD (std)",
                    "Base LSD (mean)",
                    "Base LSD (std)",
                    "LSD HF (mean)",
                    "LSD HF (std)",
                    "Base LSD HF (mean)",
                    "Base LSD HF (std)",
                    "LSD LF (mean)",
                    "LSD LF (std)",
                    "Base LSD LF (mean)",
                    "Base LSD LF (std)",
                    "RTF (mean)",
                    "RTF (std)",
                    "RTF Reciprocal (mean)",
                    "RTF Reciprocal (std)",
                ]
            )

        writer.writerow(results)


def test(args):
    def cal_snr(pred, target):
        return (
            20
            * torch.log10(
                torch.norm(target, dim=-1)
                / torch.norm(pred - target, dim=-1).clamp(min=1e-8)
            )
        ).mean()

    stft = STFTMag(2048, 512)

    def cal_lsd(pred, target, hf):
        sp = torch.log10(stft(pred).square().clamp(1e-8))
        st = torch.log10(stft(target).square().clamp(1e-8))
        return (
            (sp - st).square().mean(dim=1).sqrt().mean(),
            (sp[:, hf:, :] - st[:, hf:, :]).square().mean(dim=1).sqrt().mean(),
            (sp[:, :hf, :] - st[:, :hf, :]).square().mean(dim=1).sqrt().mean(),
        )

    hparams = OC.load("hparameter.yaml")
    hparams.save = args.save or False
    if args.cuda:
        model = NuWave2(hparams, False).cuda()
    else:
        model = NuWave2(hparams, False)

    if args.ema:
        ckpt_path = glob(
            os.path.join(hparams.log.checkpoint_dir, f"*_epoch={args.resume_from}_EMA")
        )[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    else:
        ckpt_path = glob(
            os.path.join(hparams.log.checkpoint_dir, f"*_epoch={args.resume_from}.ckpt")
        )[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
    print(ckpt_path)
    model.eval()
    model.freeze()
    os.makedirs(f"{hparams.log.test_result_dir}/{args.sr}", exist_ok=True)

    results = []
    for i in range(5):
        snr_list = []
        base_snr_list = []
        lsd_list = []
        base_lsd_list = []
        lsd_hf_list = []
        base_lsd_hf_list = []
        lsd_lf_list = []
        base_lsd_lf_list = []
        # RTF: Real Time Factor
        rtf_list = []
        t = model.test_dataloader(sr=args.sr)
        hf = int(1025 * (args.sr / hparams.audio.sampling_rate))
        for j, batch in enumerate(tqdm(t)):
            wav, wav_l, band = batch
            if args.cuda:
                # wav = wav.cuda()
                wav_l = wav_l.cuda()
                band = band.cuda()
            else:
                # wav = wav
                wav_l = wav_l
                band = band

            start_time = time.time()
            wav_up, *_ = model.inference(
                wav_l, band, 8, eval(hparams.dpm.infer_schedule)
            )
            run_time = time.time() - start_time
            rtf_list.append(run_time / wav_l.size(1) * hparams.audio.sampling_rate)

            wav_up = wav_up.cpu().detach()
            wav_l = wav_l.cpu().detach()

            snr_list.append(cal_snr(wav_up, wav))
            base_snr_list.append(cal_snr(wav_l, wav))
            lsd_j, lsd_hf_j, lsd_lf_j = cal_lsd(wav_up, wav, hf)
            base_lsd_j, base_lsd_hf_j, base_lsd_lf_j = cal_lsd(wav_l, wav, hf)
            lsd_list.append(lsd_j)
            base_lsd_list.append(base_lsd_j)
            lsd_hf_list.append(lsd_hf_j)
            base_lsd_hf_list.append(base_lsd_hf_j)
            lsd_lf_list.append(lsd_lf_j)
            base_lsd_lf_list.append(base_lsd_lf_j)

            if args.save and i == 0:
                swrite(
                    f"{hparams.log.test_result_dir}/{args.sr}/test_{j}_up.wav",
                    hparams.audio.sampling_rate,
                    wav_up[0].detach().cpu().numpy(),
                )
                swrite(
                    f"{hparams.log.test_result_dir}/{args.sr}/test_{j}_orig.wav",
                    hparams.audio.sampling_rate,
                    wav[0].detach().cpu().numpy(),
                )
                swrite(
                    f"{hparams.log.test_result_dir}/{args.sr}/test_{j}_down.wav",
                    hparams.audio.sampling_rate,
                    wav_l[0].detach().cpu().numpy(),
                )

        snr = torch.stack(snr_list, dim=0).mean()
        base_snr = torch.stack(base_snr_list, dim=0).mean()
        lsd = torch.stack(lsd_list, dim=0).mean()
        base_lsd = torch.stack(base_lsd_list, dim=0).mean()
        lsd_hf = torch.stack(lsd_hf_list, dim=0).mean()
        base_lsd_hf = torch.stack(base_lsd_hf_list, dim=0).mean()
        lsd_lf = torch.stack(lsd_lf_list, dim=0).mean()
        base_lsd_lf = torch.stack(base_lsd_lf_list, dim=0).mean()
        rtf = torch.tensor(rtf_list).mean()
        rtf_reciprocal = 1 / rtf
        dict = {
            "snr": snr.item(),
            "base_snr": base_snr.item(),
            "lsd": lsd.item(),
            "base_lsd": base_lsd.item(),
            "lsd_hf": lsd_hf.item(),
            "base_lsd_hf": base_lsd_hf.item(),
            "lsd_lf": lsd_lf.item(),
            "base_lsd_lf": base_lsd_lf.item(),
            "rtf": rtf.item(),
            "rtf_reciprocal": rtf_reciprocal.item(),
        }
        results.append(
            torch.stack(
                [
                    snr,
                    base_snr,
                    lsd,
                    base_lsd,
                    lsd_hf,
                    base_lsd_hf,
                    lsd_lf,
                    base_lsd_lf,
                    rtf,
                    rtf_reciprocal,
                ],
                dim=0,
            ).unsqueeze(-1)
        )
        print(dict)

    # Save results to csv
    # Loop the results and calculate mean and std of results
    results = torch.cat(results, dim=1)
    # Get mean and std in [[mean, std], [mean, std], ...] format
    results = torch.stack([results.mean(dim=1), results.std(dim=1)], dim=1)
    # Convert to [mean, std, mean, std]
    results = results.flatten().tolist()
    # Add sample rate to the beginning of the list
    results.insert(0, args.sr)

    if args.resume_from == 629:
        save_results_to_csv(results, "results_48kHz.csv")
    elif args.resume_from == 584:
        save_results_to_csv(results, "results_16kHz.csv")
    else:
        save_results_to_csv(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume_from",
        type=int,
        required=True,
        help="Resume Checkpoint epoch number",
    )
    parser.add_argument(
        "-e",
        "--ema",
        action="store_true",
        required=False,
        help="Start from ema checkpoint",
    )
    parser.add_argument("--save", action="store_true", required=False, help="Save file")
    parser.add_argument(
        "--cuda", action="store_true", required=True, help="Enable CPU running"
    )
    parser.add_argument("--sr", type=int, required=True, help="input sampling rate")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    test(args)
