#!/usr/bin/env python
# graph.py : csvを画像化するスクリプト

import pandas as pd
import pathlib
import pprint
from matplotlib import pyplot as plt
import tqdm
import numpy as np

def plot_per_orgdata(input_dir, out_dir):
    """
    csv波形を画像化する処理。元波形単位でプロットする。

    Args:
        input_dir (str): csv波形格納ディレクトリ。配下が以下のフォルダ・ファイル構成となっている必要がある。
                            ${input_dir}/{クラスラベル}/{元波形名}/*_{センサ名}_ ... _{拡張番号}.csv
        out_dir (str): 画像出力先ディレクトリ
    """
    path = pathlib.Path(input_dir)

    # センサリスト
    sens_list = ["sensor1", "sensor2", "sensor3"]

    # クラスラベルのループ
    clabel_list = sorted(path.glob('*'))
    for clabel in tqdm.tqdm(clabel_list):

        # 元波形のループ
        org_list = sorted(clabel.glob('*'))
        for org in tqdm.tqdm(org_list):

            # データ拡張波形のリストを取得
            file_list = sorted(org.glob('*.csv'))
            if len(file_list)<1: # フォルダによっては空かもしれないので
                continue

            # センサ名とデータ拡張番号のみフォーマットできるように、先頭のファイル名を{}で置換
            file_head = file_list[0]
            tok = file_head.name.split("_")
            sens = tok[ 1]  # 2番目がセンサ
            num  = tok[-1]  # 末尾が拡張番号
            formatter = file_head.name.replace(sens, "{}").replace(num, "{}.csv") # TODO: ２つ置換するものがあったら意図通り動かんね汗

            fig = plt.figure()
            fig.suptitle(org)
            plt.tight_layout()

            # 拡張番号のループ
            # memo: 拡張数は元波形毎に異なるのでwhileで処理
            counter = 0
            while(True):
                file_not_exist_flag = False

                # センサ種のループ
                for idx, sens in enumerate(sens_list):

                    # formatterを用いてファイル名を作成
                    csv_file = formatter.format(sens, counter)

                    # pathに合成
                    csv_path = org.joinpath(csv_file)

                    # この拡張番号があるかどうかをチェック
                    if csv_path.exists()==False:
                        file_not_exist_flag = True
                        break
                    
                    # csv読み込み
                    dataframe = pd.read_csv(csv_path, header=None)

                    # プロット
                    plt.subplot(len(sens_list), 1, idx+1)
                    plt.plot(dataframe.index, dataframe[0].values, linewidth=0.3)
                    if counter < 2: # 高速化
                        plt.grid(True, linestyle=":")

                    # yのレンジを合わせるため、最大最小の中間を計算
                    center_val = (np.max(dataframe[0].values) + np.min(dataframe[0].values))/2
                    max_val = 0
                    min_val = 0

                    # センサ毎にレンジを固定
                    if sens == "sensor1":
                        max_val = center_val + 2000
                        min_val = center_val - 2000
                    elif sens == "sensor2":
                        max_val = center_val + 800
                        min_val = center_val - 800
                    elif sens == "sensor3":
                        max_val = center_val + 1
                        min_val = center_val - 1

                    if counter < 2: # 高速化
                        plt.ylim([min_val, max_val])
                        # plt.xlabel("samples")
                        plt.ylabel(sens)

                # csvファイルが存在しなかった場合
                if file_not_exist_flag == True:
                    if counter == 0: # 通し番号が1からスタートするデータがあるので、もう一度チャンスをあげる。
                        counter = counter + 1
                        continue
                    else:
                        break # while終了

                counter = counter + 1
                if counter == 30: # 最大プロット数(任意パラメータ)
                    break # while終了
            
            # png出力
            out_dir = pathlib.Path(out_dir)
            #png_path = out_dir.joinpath(org).with_suffix(".png")
            png_path = out_dir.joinpath(clabel.name + "_" + org.name).with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            plt.xlabel("samples")
            plt.savefig(png_path, dpi=300)
            plt.clf()

    return

def augmentation(input_dir, out_dir):
    """
    旧版拡張データを周波数拡張データに変換する。
    拡張数は旧版拡張に合わせる。
    周波数拡張の元データは、旧版拡張の真ん中の拡張番号を用いる。

    Args:
        input_dir (str): 旧版拡張csvデータ格納ディレクトリ。
                         配下が以下のフォルダ・ファイル構成となっている必要がある。
                           ${input_dir}/{クラスラベル}/{元波形名}/*_{センサ名}_ ... _{拡張番号}.csv
        out_dir   (str): 周波数拡張csvデータ格納ディレクトリ。
                         配下が以下のフォルダ・ファイル構成となる。input_dir側と同じ構成となる。
                           ${out_dir}/{クラスラベル}/{元波形名}/*_{センサ名}_ ... _{拡張番号}.csv
    """
    path = pathlib.Path(input_dir)

    # センサリスト
    sens_list = ["sensor1", "sensor2", "sensor3"]

    strength = {"sensor1": 0.8, "sensor2": 0.8, "sensor3":0.3}

    #-----------------------------------
    # まずは旧版の拡張数を計算する
    #-----------------------------------

    # 配下のcsvリスト全てを取得
    file_list = sorted(path.glob('**/*.csv'))
    aug_num_list = {}
    for f in file_list:
        org_name = f.parent.name
        clabel   = f.parent.parent.name

        # 各クラス・元波形名毎にcsvファイル数をカウント
        if clabel + "/" + org_name in aug_num_list:
            aug_num_list[clabel + "/" + org_name] = aug_num_list[clabel + "/" + org_name] + 1
        else:
            aug_num_list[clabel + "/" + org_name] = 1

    # センサはそれぞれ別のcsvファイルであるため、個数をセンサ数で割って削減
    for k, v in aug_num_list.items():
        aug_num_list[k] = int(v/len(sens_list))

    # 整形して出力
    print(pprint.pprint(aug_num_list))

    # 元波形毎のループ
    for k,v in tqdm.tqdm(aug_num_list.items()):

        org_path = path.joinpath(k)
        aug_num = v

        print(org_path)

        # データ拡張波形のリストを取得
        file_list = sorted(org_path.glob('*.csv'))
        if len(file_list)<1: # フォルダによっては空かもしれないので
            continue

        # センサ名とデータ拡張番号のみフォーマットできるように、先頭のファイル名を{}で置換
        file_head = file_list[0]
        tok = file_head.name.split("_")
        sens = tok[ 1]  # 2番目がセンサ
        num  = tok[-1]  # 末尾が拡張番号
        formatter = file_head.name.replace(sens, "{}").replace(num, "{}.csv") # TODO: ２つ置換するものがあったら意図通り動かんね汗

        # センサ種のループ
        for sens in sens_list:
            csv_file = formatter.format(sens, int(aug_num/2)) # 拡張番号の中央を周波数変換拡張に使用
            input_path = org_path.joinpath(csv_file)

            # 入力を読み込み
            dataframe = pd.read_csv(input_path, header=None)

            # データ拡張
            if sens == sens_list[0]: # 先頭のセンサでシードを設定
                aug_data = data_augmentation_fft_perturbation(dataframe[0].values, augmentation_num=aug_num, strength=strength[sens], seed=777)
            else:
                aug_data = data_augmentation_fft_perturbation(dataframe[0].values, augmentation_num=aug_num, strength=strength[sens])

            # 拡張数のループ
            for aug_cnt in tqdm.tqdm(range(aug_num)):

                # 出力ファイル名
                csv_file = formatter.format(sens, aug_cnt)

                clabel   = org_path.parent.name
                org_name = org_path.name

                # path合成とフォルダ作成
                out_path = pathlib.Path(out_dir)
                out_path = out_path.joinpath(clabel, org_name, csv_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # csv出力
                df = pd.DataFrame(aug_data[aug_cnt,:])
                df.to_csv(out_path, header=None, index=None)

def data_augmentation_fft_perturbation(input_wave, augmentation_num=10, strength=0.3, seed=None, except_low_bin=3):
    """
    データ拡張。周波数領域で振幅・位相スペクトルそれぞれに摂動をかける。

    Args:
        input_wave (numpy.ndarray): 入力波形
        augmentation_num (int): 拡張数
        seed (int): bin選択および摂動向けの乱数のシード
    Returns:
        (numpy.ndarray): 拡張波形、shape == (拡張数, 入力波形サイズ)
    """
    from scipy import fftpack

    # シード初期化
    if seed is not None:
        np.random.seed(seed)
    
    # 入力データは偶数のみを考慮
    assert(len(input_wave)%2 == 0)

    # parameters
    perturb_bin_rate = 0.20 # 20%のbinに摂動をかける
    perturb_strength_amp   = strength
    perturb_strength_phase = strength

    # FFT(DFT)
    F = fftpack.fft(input_wave)

    # 複素⇒振幅、位相に分解
    amplitude = np.absolute(F)
    phase = np.angle(F)

    # # for debug
    # plt.figure(figsize=(16.0, 6.0))
    # plt.subplot(3,1,1); plt.plot(range(len(input_wave))           , input_wave                     , linewidth=1)
    # plt.subplot(3,1,2); plt.plot(range(len(amplitude[1:]))        , np.log10(amplitude[1:])        , linewidth=1)
    # plt.subplot(3,1,3); plt.plot(range(len(phase[1:]))            , phase[1:]                      , linewidth=1)

    # メモリ確保
    out_waves = np.zeros((augmentation_num,len(input_wave)), dtype=input_wave.dtype)

    # 拡張数のループ
    for i in range(augmentation_num):

        # 振幅側の摂動
        perturb_bin_num = int((len(amplitude)/2-1 - except_low_bin) * perturb_bin_rate)
        perturb_idx = np.random.randint(1 + except_low_bin, int(len(amplitude)/2), perturb_bin_num)
        amplitude_perturb = amplitude.copy()
        for idx in perturb_idx:
            perturb = np.random.normal(loc=0, scale=perturb_strength_amp, size=1)
            amplitude_perturb[ idx] = amplitude[ idx] + np.power(10, perturb[0])
            amplitude_perturb[-idx] = amplitude[-idx] + np.power(10, perturb[0])
            
        # 位相側の摂動
        perturb_bin_num = int((len(phase)/2-1 - except_low_bin) * perturb_bin_rate)
        perturb_idx = np.random.randint(1 + except_low_bin, int(len(phase)/2), perturb_bin_num)
        phase_perturb = phase.copy()
        for idx in perturb_idx:
            perturb = np.random.normal(loc=0, scale=perturb_strength_phase, size=1)
            phase_perturb[ idx] = phase[ idx] + perturb[0]
            phase_perturb[-idx] = phase[-idx] - perturb[0]

        # 振幅、位相⇒複素に結合
        F_perturb = amplitude_perturb * (np.cos(phase_perturb) + np.sin(phase_perturb) * 1j)

        # IFFT
        out_wave = fftpack.ifft(F_perturb).real

        # 出力用配列に複製して格納
        out_waves[i,:] = out_wave.copy()

        # # for debug
        # plt.subplot(3,1,1); plt.plot(range(len(out_wave))             , out_wave                       , linewidth=1)
        # plt.subplot(3,1,2); plt.plot(range(len(amplitude_perturb[1:])), np.log10(amplitude_perturb[1:]), linewidth=1)
        # plt.subplot(3,1,3); plt.plot(range(len(phase_perturb[1:]))    , phase_perturb[1:]              , linewidth=1)
    # # for debug
    # plt.savefig("sample.png")

    return out_waves

if __name__ == "__main__":

    # 元波形のプロット
    plot_per_orgdata("./300p", "./png/300p")
    
    # データ拡張メイン
    augmentation("../dataset/300p", "../dataset/300p_aug_freq")
    plot_per_orgdata("../dataset/300p_aug_freq", "../dataset/png/300p_aug_freq") # 周波数拡張データのプロット

    # データ拡張メイン
    augmentation("../dataset/300p", "../dataset/300p_aug_freq_excep")
    plot_per_orgdata("../dataset/300p_aug_freq_excep", "../dataset/png/300p_aug_freq_excep") # 周波数拡張データのプロット
