from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch
from catalyst.dl import Runner
from nyaggle.feature_store import cached_feature, load_features, load_feature
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from src.models import Model
from src.utils import get_folds, seed_everything, get_timestamp


def run(
    X_seq_train, X_cont_train, y_train, X_seq_test, X_cont_test, timestamp, random_state
):
    seed_everything(random_state)

    oof_preds = np.zeros(len(X_seq_train))
    test_preds = np.zeros(len(X_seq_test))
    cv_scores = []
    for i, (trn_idx, val_idx) in enumerate(
        get_folds(5, "stratified", random_state).split(X_cont_train, y_train)
    ):
        print(f"fold {i + 1}")
        train_dataset = TensorDataset(
            torch.from_numpy(X_seq_train[trn_idx]).float(),
            torch.from_numpy(X_cont_train[trn_idx]).float(),
            torch.from_numpy(y_train[trn_idx]).float(),
        )
        valid_dataset = TensorDataset(
            torch.from_numpy(X_seq_train[val_idx]).float(),
            torch.from_numpy(X_cont_train[val_idx]).float(),
            torch.from_numpy(y_train[val_idx]).float(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_seq_test).float(), torch.from_numpy(X_cont_test).float()
        )

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
        valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=128)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)
        loaders = {"train": train_loader, "valid": valid_loader}

        runner = CustomRunner(device="cuda")

        model = Model(
            in_channels=X_seq_train.shape[1],
            n_cont_features=X_cont_train.shape[1],
            hidden_channels=64,
            kernel_sizes=[3, 5, 7, 15, 21, 51, 101],
            out_dim=1,
        )
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-6
        )

        logdir = f"./logdir/{timestamp}_fold{i}"
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=30,
            verbose=True,
        )

        pred = np.concatenate(
            list(
                map(
                    lambda x: x.cpu().numpy(),
                    runner.predict_loader(
                        loader=valid_loader,
                        resume=f"{logdir}/checkpoints/best.pth",
                        model=model,
                    ),
                )
            )
        )
        oof_preds[val_idx] = pred
        score = average_precision_score(y_train[val_idx], pred)
        cv_scores.append(score)
        print("score", score)

        pred = np.concatenate(
            list(
                map(
                    lambda x: x.cpu().numpy(),
                    runner.predict_loader(
                        loader=test_loader,
                        resume=f"{logdir}/checkpoints/best.pth",
                        model=model,
                    ),
                )
            )
        )
        test_preds += pred / 5
    return oof_preds, test_preds, cv_scores


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        seq, cont, y = batch
        pred = self.model(seq, cont)
        loss = self.criterion(pred, y)
        self.batch_metrics = {"loss": loss}
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = self._batch2device(batch, self.device)
        if len(batch) == 2:
            seq, cont = batch
        elif len(batch) == 3:
            seq, cont, _ = batch
        else:
            raise RuntimeError
        pred = self.model(seq, cont)
        return pred


@cached_feature("pad_spec")
def create_pad_spectrum(df: pd.DataFrame, spec: pd.DataFrame):
    spec = spec.copy()
    spec["wave_index"] = spec.groupby("spectrum_filename").intensity.transform(
        lambda x: np.arange(len(x))
    )
    feat = pd.pivot(
        spec, index="spectrum_filename", columns="wave_index", values="intensity"
    ).ffill(axis=1)
    feat.columns = [f"intensity_{i:03d}" for i in range(512)]
    df = df.merge(feat, left_on="spectrum_filename", right_index=True)
    return df.iloc[:, -len(feat.columns) :]


if __name__ == "__main__":
    submission = pd.read_csv("input/atmaCup5__sample_submission.csv")
    train = pd.read_csv("input/train.csv")
    all_df = load_feature("all", "working")
    spec = load_feature("spec", "working")
    pad_spec = create_pad_spectrum(all_df, spec)

    # add derivative spectra
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    spec_array = np.stack(
        [
            pad_spec.values,
            scipy.signal.savgol_filter(pad_spec, 5, 2, deriv=0, axis=1),
            scipy.signal.savgol_filter(pad_spec, 5, 2, deriv=1, axis=1),
            scipy.signal.savgol_filter(pad_spec, 5, 2, deriv=2, axis=1),
        ],
        axis=1,
    )  # (14388, 4, 512)

    # sample-wise scaling
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116#442741
    spec_array /= spec_array.std(axis=2).reshape(-1, 4, 1)
    X_seq_train = spec_array[: len(train)]
    X_seq_test = spec_array[len(train) :]

    # continuous features
    data = load_features(
        all_df,
        feature_names=[
            "fitting",
            "peak_around",
            "intensity_stats",
            "savgol_peak",
            "fitting_combination",
        ],
        ignore_columns=["spectrum_id", "spectrum_filename", "chip_id"],
    )
    train = data[data.target.notnull()].copy()
    test = data[data.target.isnull()].copy()
    target_col = "target"
    drop_cols = ["spectrum_id", "spectrum_filename", "chip_id"]
    X_train = train.drop(drop_cols + [target_col], axis=1)
    y_train = train[target_col].values
    X_test = test.drop(drop_cols + [target_col], axis=1)

    # fill inf/nan
    X_train.replace(np.inf, np.nan, inplace=True)
    X_test.replace(np.inf, np.nan, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_train.mean(), inplace=True)

    # rankgauss transform
    # https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
    prep = QuantileTransformer(output_distribution="normal")
    X_cont_train = prep.fit_transform(X_train)
    X_cont_test = prep.transform(X_test)

    # train and predict
    timestamp = get_timestamp()
    oof_preds, test_preds, cv_scores = run(
        X_seq_train,
        X_cont_train,
        y_train,
        X_seq_test,
        X_cont_test,
        timestamp,
        random_state=0,
    )

    # save results
    print(cv_scores)
    output_dir = Path(f"./output/{timestamp}")
    output_dir.mkdir(parents=True)
    pd.DataFrame(oof_preds).to_csv(output_dir / f"{timestamp}_oof.csv", index=False)
    submission["target"] = test_preds
    submission.to_csv(output_dir / f"{timestamp}_submission.csv", index=False)
