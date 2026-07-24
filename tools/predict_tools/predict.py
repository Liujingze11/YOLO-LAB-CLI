import sys
from dataclasses import dataclass
from pathlib import Path
import json

# ensure project root on sys.path when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ultralytics import YOLO
from cli_config import PREDICT_DIR, BEST_SEG_MODEL, TEST_IMAGES_DIR

_TOOLS_DIR = Path(__file__).resolve().parent

# =========================
# 只改这里：通用参数
# =========================
@dataclass
class InferConfig:
    model_path: str = BEST_SEG_MODEL
    source: str = TEST_IMAGES_DIR
    save_dir: str = str(Path(PREDICT_DIR) / "predict_result")

    conf: float = 0.406
    imgsz: int = 640

    # 外置任务参数文件
    task_param_file: str = str(_TOOLS_DIR / "infer_task_params.json")

    # 输出文件后缀
    out_suffix: str = "_overlay.jpg"


class TaskParamLoader:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.params = self._load_json()

    def _load_json(self) -> dict:
        if not self.json_path.exists():
            raise FileNotFoundError(f"找不到任务参数文件: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_task_params(self, task: str) -> dict:
        if task not in self.params:
            raise KeyError(f"配置文件里没有 task={task} 的参数")
        return self.params[task]


class YOLOInferencer:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.model = YOLO(self.cfg.model_path)
        self.task_loader = TaskParamLoader(self.cfg.task_param_file)

        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.task = self._detect_task()
        self.task_params = self.task_loader.get_task_params(self.task)

    def _detect_task(self) -> str:
        task = getattr(self.model, "task", None)
        if not task:
            raise ValueError("无法从模型中识别 task")
        return task

    def _build_predict_kwargs(self) -> dict:
        kwargs = {
            "source": self.cfg.source,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf,
            "save": False
        }
        task_predict_kwargs = self.task_params.get("predict", {})
        kwargs.update(task_predict_kwargs)
        return kwargs

    def _build_plot_kwargs(self) -> dict:
        return self.task_params.get("plot", {})

    def run(self):
        print(f"模型: {self.cfg.model_path}")
        print(f"自动识别任务: {self.task}")
        print(f"输入源: {self.cfg.source}")
        print(f"输出目录: {self.save_dir}")

        predict_kwargs = self._build_predict_kwargs()
        plot_kwargs = self._build_plot_kwargs()

        results = self.model.predict(**predict_kwargs)

        for i, r in enumerate(results):
            if getattr(r, "path", None):
                stem = Path(r.path).stem
            else:
                stem = f"result_{i:05d}"

            out_path = self.save_dir / f"{stem}{self.cfg.out_suffix}"
            r.save(filename=str(out_path), **plot_kwargs)

        print(f"推理完成，共保存 {len(results)} 张结果到: {self.save_dir}")


if __name__ == "__main__":
    cfg = InferConfig(
        model_path=BEST_SEG_MODEL,
        source=TEST_IMAGES_DIR,
        save_dir=str(Path(PREDICT_DIR) / "predict_result"),
        conf=0.406,
        imgsz=640,
        task_param_file=str(_TOOLS_DIR / "infer_task_params.json"),
        out_suffix="_overlay.jpg"
    )

    inferencer = YOLOInferencer(cfg)
    inferencer.run()
