from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from nyaggle.feature_store import cached_feature
from tqdm import tqdm


@cached_feature("all", "working")
def create_all_df(train, test):
    """trainとtestをくっつけたDataFrameを作る"""
    print("prepare all_df")
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    return all_df


@cached_feature("spec", "working")
def create_spec_df(spec_dir: Path):
    """スペクトルデータをひとつのDataFrameにまとめる"""
    print("prepare spec_df")
    dfs = []
    for path in tqdm(all_df.spectrum_filename):
        spec_df = pd.read_csv(spec_dir / path, sep="\t", header=None)
        spec_df.columns = ["wavelength", "intensity"]
        spec_df["spectrum_filename"] = path
        dfs.append(spec_df)
    spec = pd.concat(dfs, ignore_index=True, sort=False)
    spec.sort_values(["spectrum_filename", "wavelength"], inplace=True)
    spec.reset_index(drop=True, inplace=True)
    return spec


@cached_feature("spec_peak_around", "working")
def get_peak_around(spec):
    """ピーク周りの波形を取り出す"""
    print("prepare peak_around")

    def inner(x):
        i = np.argmax(x)
        return pd.Series(x[i - 10 : i + 10])

    peak_around = spec.groupby("spectrum_filename").intensity.apply(
        lambda x: inner(x.values)
    )
    peak_around = peak_around.unstack()
    peak_around.columns = [f"peak_around_{i:02d}" for i in peak_around.columns]
    return peak_around.reset_index()


@cached_feature("intensity_stats")
def create_intensity_stats(df, spec):
    """スペクトル全体の基本統計量を取る"""
    print("create instensity_stats")
    key = "spectrum_filename"
    feat = spec.groupby(key).intensity.agg(
        ["min", "max", "mean", "std", "median", scipy.stats.skew, scipy.stats.kurtosis]
    )
    feat.columns = "intensity_" + feat.columns
    df = df.merge(feat, how="left", left_on=key, right_index=True)
    return df.iloc[:, -len(feat.columns) :]


@cached_feature("peak_around")
def create_peak_around_feature(df, peak_around):
    """ピークまわりの基本統計量を取る"""
    print("create peak_around feature")
    peak = peak_around.set_index("spectrum_filename")
    feat = pd.DataFrame(index=peak.index)
    feat["peak_around_max"] = peak.max(axis=1)
    feat["peak_around_min"] = peak.min(axis=1)
    feat["peak_around_mean"] = peak.mean(axis=1)
    feat["peak_around_median"] = peak.median(axis=1)
    feat["peak_around_std"] = peak.std(axis=1)
    feat["peak_around_skew"] = peak.apply(scipy.stats.skew, axis=1)
    feat["peak_around_kurt"] = peak.apply(scipy.stats.kurtosis, axis=1)
    feat["peak_around_max_per_mean"] = (
        feat["peak_around_max"] / feat["peak_around_mean"]
    )
    df = df.merge(feat, how="left", left_on="spectrum_filename", right_index=True)
    return df.iloc[:, -len(feat.columns) :]


@cached_feature("fitting")
def create_fitting(df, fitting):
    """fittingをmainに結合する"""
    print("create fitting")
    df = df.merge(fitting, how="left", on="spectrum_id")
    return df[fitting.columns].drop("spectrum_id", axis=1)


@cached_feature("fitting_combination")
def create_fitting_combination(df, fitting):
    """fittingに対して四則演算する"""
    print("create fitting_combination feature")
    df = df.merge(fitting, how="left", on="spectrum_id")
    df["params1_div_params3"] = df["params1"] / df["params3"]
    df["params1_div_params4"] = df["params1"] / df["params4"]
    df["params2_sub_params5"] = df["params2"] - df["params5"]
    df["params3_sub_params6"] = df["params3"] - df["params6"]
    df["params3_div_params6"] = df["params3"] / df["params6"]
    df["params4_div_params6"] = df["params4"] / df["params6"]
    return df.iloc[:, -6:]


@cached_feature("savgol_peak")
def create_savgol_peak(df, spec):
    """Savitzky-Golay Filteringしたスペクトルに対して統計量を取る"""
    print("create savgol_peak feature")
    feat = pd.DataFrame(index=df.spectrum_filename.unique())
    for name, sp in tqdm(
        spec.set_index("wavelength").groupby("spectrum_filename").intensity
    ):
        sp = sp.to_frame()
        for deriv in range(3):
            sp["filtered"] = scipy.signal.savgol_filter(sp.intensity, 5, 2, deriv=deriv)
            feat.loc[name, f"savgol_{deriv}_peak_max"] = sp.filtered.max()
            feat.loc[name, f"savgol_{deriv}_peak_min"] = sp.filtered.min()
            feat.loc[name, f"savgol_{deriv}_peak_idxmax"] = sp.filtered.idxmax()
            feat.loc[name, f"savgol_{deriv}_peak_idxmin"] = sp.filtered.idxmin()
            for i in range(-5, 5 + 1):
                if i == 0:
                    continue
                else:
                    feat.loc[
                        name, f"savgol_{deriv}_peak_pct_change_{i}"
                    ] = sp.filtered.pct_change(i)[sp.filtered.idxmax()]
            feat.loc[name, f"savgol_{deriv}_peak_range"] = (
                feat.loc[name, f"savgol_{deriv}_peak_max"]
                - feat.loc[name, f"savgol_{deriv}_peak_min"]
            )
            feat.loc[name, f"savgol_{deriv}_peak_idxrange"] = np.abs(
                feat.loc[name, f"savgol_{deriv}_peak_idxmax"]
                - feat.loc[name, f"savgol_{deriv}_peak_idxmin"]
            )
    df = df.merge(feat, how="left", left_on="spectrum_filename", right_index=True)
    return df.iloc[:, -len(feat.columns) :]


@cached_feature("spec_percentile")
def create_spec_percentile(df, spec):
    """最大ピークに対してn%の高さで切り、その高さでのピークの個数や、ピークの存在する領域幅を取る"""
    print("create spec_percentile feature")
    spec = spec.copy()
    spec["intensity"] -= spec.groupby("spectrum_filename").intensity.transform("min")
    for p in tqdm(range(1, 20)):
        spec[f"percentile_{p*5}"] = spec["intensity"] > spec.groupby(
            "spectrum_filename"
        ).intensity.transform(lambda x: x.max() * p * 0.05)
    feat = pd.DataFrame(index=spec.spectrum_filename.unique())
    group = spec.set_index("wavelength").groupby("spectrum_filename")
    for p in tqdm(range(1, 20)):
        feat[f"peak_percentile_{p*5}"] = group[f"percentile_{p*5}"].sum()
        feat[f"peak_count_{p*5}"] = group[f"percentile_{p*5}"].apply(
            lambda x: (x.astype(int).diff() > 0).sum()
        )
    df = df.merge(feat, how="left", left_on="spectrum_filename", right_index=True)
    return df.iloc[:, -len(feat.columns) :]


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent
    INPUT_DIR = ROOT_DIR / "input"
    SPEC_DIR = INPUT_DIR / "spectrum_raw"
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    fitting = pd.read_csv(INPUT_DIR / "fitting.csv")
    submission = pd.read_csv(INPUT_DIR / "atmaCup5__sample_submission.csv")

    all_df = create_all_df(train, test)
    spec = create_spec_df(SPEC_DIR)
    peak_around = get_peak_around(spec)

    create_peak_around_feature(all_df, peak_around)
    create_intensity_stats(all_df, spec)
    create_fitting(all_df, fitting)
    create_fitting_combination(all_df, fitting)
    create_savgol_peak(all_df, spec)
    create_spec_percentile(all_df, spec)
