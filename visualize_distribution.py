import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams["font.family"] = "Times New Roman"  # 全局字体


def get_kaf_frequencies():
    return np.array(
        [
            1,  # unknow
            20999,  # fixed
            321,  # revolute free
            3058,  # revolute controlled
            17,  # revolute static
            34,  # prismatic free
            3058,  # prismatic controlled
            6,  # prismatic static
            3,  # spherical free
            6,  # spherical controlled
            6,  # spherical static
            3107,  # supported
            596,  # flexible
            1348,  # unrelated
        ]
    )


def get_func_frequencies():
    return np.array(
        [
            7977,  # other
            7329,  # handle
            8067,  # housing
            31730,  # support
            13100,  # frame
            1934,  # button
            256,  # wheel
            2266,  # display
            3499,  # cover
            155,  # plug
            123,  # port
            10651,  # door
            9375,  # container
        ]
    )


def plot_sorted_bars(data, labels, title, filename_prefix, zoom_threshold_ratio=0.1):
    # 排序
    sorted_idx = np.argsort(data)[::-1]  # 从大到小
    sorted_data = data[sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]

    # 颜色映射
    cmap = plt.cm.get_cmap("tab20", len(sorted_data))
    colors = [cmap(i) for i in range(len(sorted_data))]

    def format_axes(ax):
        """统一坐标轴格式"""
        # 科学记数法
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)  # 启用科学记数法
        formatter.set_powerlimits((-2, 2))  # 控制何时切换成科学记数法
        ax.yaxis.set_major_formatter(formatter)
        offset = ax.yaxis.get_offset_text()
        offset.set_fontsize(24)
        offset.set_va("bottom")
        offset.set_ha("left")
        offset.set_x(-0.05)
        offset.set_y(1.02)

        # 字体大小
        ax.tick_params(axis="both", labelsize=24)
        ax.set_ylabel("Frequency", fontsize=24)
        ax.set_xlabel("", fontsize=24)

        # 只保留左下坐标轴
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ==== 完整图 ====
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(sorted_data)), sorted_data, color=colors, width=1)
    ax.set_xticks(range(len(sorted_data)))
    ax.set_xticklabels(sorted_labels, rotation=60, ha="right", fontsize=24)

    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_full.png", dpi=300)
    plt.close()

    # ==== 小值局部放大 ====
    threshold = sorted_data.max() * zoom_threshold_ratio
    mask_small = sorted_data < threshold
    if np.any(mask_small):
        small_data = sorted_data[mask_small]
        small_labels = [
            sorted_labels[i] for i in range(len(sorted_data)) if mask_small[i]
        ]
        small_colors = [colors[i] for i in range(len(sorted_data)) if mask_small[i]]

        fig_width = (len(small_data) + 1.5) / (len(sorted_data) + 1.5) * 8
        fig, ax = plt.subplots(figsize=(fig_width, 4))
        bars = ax.bar(range(len(small_data)), small_data, color=small_colors, width=1)
        ax.set_xticks(range(len(small_data)))
        ax.set_xticklabels(small_labels, rotation=60, ha="right", fontsize=24)

        for bar, val in zip(bars, small_data):
            ax.text(
                bar.get_x() + bar.get_width() / 2 + 0.3,
                val,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=18,
                rotation=60,
            )

        format_axes(ax)
        ax.set_ylabel("", fontsize=24)
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_zoom.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    kaf_labels = [
        "unknow",
        "fixed",
        "rev free",
        "rev ctrl",
        "rev static",
        "pri free",
        "pri ctrl",
        "pri static",
        "sph free",
        "sph ctrl",
        "sph static",
        "supported",
        "flexible",
        "unrelated",
    ]
    kaf_data = get_kaf_frequencies()
    plot_sorted_bars(
        kaf_data, kaf_labels, "KAF Frequencies", "kaf", zoom_threshold_ratio=0.1
    )

    func_labels = [
        "other",
        "handle",
        "housing",
        "support",
        "frame",
        "button",
        "wheel",
        "display",
        "cover",
        "plug",
        "port",
        "door",
        "container",
    ]
    func_data = get_func_frequencies()
    plot_sorted_bars(
        func_data, func_labels, "Function Frequencies", "func", zoom_threshold_ratio=0.1
    )

    print("图已保存为 kaf_full.png, kaf_zoom.png, func_full.png, func_zoom.png")
