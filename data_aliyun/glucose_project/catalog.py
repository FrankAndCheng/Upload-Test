"""Entry point -> immutable historical implementation -> existing artifacts."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Workflow:
    module: str
    source: str
    description: str
    artifacts: str
    arguments: str = "none"


WORKFLOWS = {
    "environment": Workflow("environment.check", "review/check_environment.py",
        "检查base导入与小样本训练", "review/environment_check.json"),
    "original-algorithms": Workflow("experiments.external_baselines", "review/external_b_comparison/evaluate_frozen.py",
        "恢复原六模型、重拟合并评估B", "review/external_b_comparison/；paper.ipynb"),
    "sequential": Workflow("experiments.sequential", "review/sequential_experiments/run_experiments.py",
        "按顺序执行特征消融、联合搜索、树融合与小网络实验", "review/sequential_experiments/"),
    "neural-features": Workflow("ensembles.refine_neural", "review/sequential_experiments/refine_nn_features.py",
        "训练OOF驱动的去乙肝小网络补充对照", "review/sequential_experiments/"),
    "neural-blend": Workflow("ensembles.blend_neural", "review/sequential_experiments/complete_nn_blends.py",
        "按既定权重网格补充小网络融合", "review/sequential_experiments/"),
    "preprocessing": Workflow("preprocessing.ablation", "review/preprocessing_ablation/run_ablation.py",
        "十模型标准化/PCA四组配对实验", "review/preprocessing_ablation/"),
    "predict": Workflow("inference.predict", "review/sequential_experiments/predict_selected.py",
        "加载已冻结四模型，只预测不训练", "--output指定CSV", "predict"),
    "verify-preprocessing": Workflow("validation.preprocessing", "review/preprocessing_ablation/verify_results.py",
        "重算指标并重新加载建议模型核验", "review/preprocessing_ablation/verification.json"),
    "report-sequential": Workflow("reports.sequential", "review/sequential_experiments/render_results.py",
        "读取顺序实验结果并生成报告/notebook", "review/sequential_experiments/report.md；paper_experiments.ipynb"),
    "report-preprocessing": Workflow("reports.preprocessing", "review/preprocessing_ablation/render_results.py",
        "读取预处理结果并生成报告/notebook", "review/preprocessing_ablation/report.md；paper_preprocessing_experiments.ipynb"),
    "report-historical": Workflow("reports.historical", "review/optimization/analyze_results.py",
        "早期协议的事后误差分析；会重拟合部分参考模型", "review/optimization/"),
    "legacy-optimization": Workflow("legacy.optimization", "paper_optimized.py",
        "早期1561条验证或仅原train的A/B协议", "review/optimization/", "legacy"),
    "legacy-baseline": Workflow("legacy.baseline", "review/baseline_check.py",
        "早期独立Ridge基线检查，不属于当前融合选型", "review/baseline_results.json"),
}
