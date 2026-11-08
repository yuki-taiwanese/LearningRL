# Roboto Origin 兩桌搬箱任務

這是課程 LearningRL 的新增任務。使用原始 RPO 23 關節模型，不替換成 Atlas 或 G1。
目前交付的是程式、CPU 檢查與場景預覽；沒有訓練好的 policy，也沒有宣稱已在 Isaac Lab 成功搬運。

## 任務與場景

機器人面向 +X；+Y 為機器人左側。藍桌在左，綠桌在右。
從左桌站位前方 1 m 出發，走近、雙臂接觸、抬起、抱箱側向走到右桌、放下、鬆手。
兩張桌中心相距 1.3 m，桌高 0.72 m，桌面 0.70 × 0.64 m。
箱子外部 0.24 × 0.30 × 0.20 m，厚度 0.012 m，空箱質量 0.50 kg。
這些是可調的虛擬課堂場景參數，不是手冊量測值；集中在 `layout.py`。

RPO 沒有可控制手指；本任務使用原廠 URDF 的兩個 elbow_pitch 前臂碰撞形狀夾抱箱子。
碰撞中心 (0.14, ∓0.02, 0) m 取自鎖定版本 rpo.urdf，不使用無碰撞的 elbow_yaw 當抓點。
箱子有五片面板、開口朝上，是一個受重力作用的剛體；沒有黏附、瞬移或自動抓取約束。

## 獎勵與失敗

| 階段 | 轉換條件 | 一次性分數 |
|---|---|---:|
| 走近 | 到左桌站位 10 cm 內，平移速度 <0.20 m/s，維持 0.25 s | +5 |
| 接觸 | 兩前臂各與箱接觸 >0.5 N，且靠近對應抓點，維持 0.15 s | +5 |
| 抬起 | 箱底高於桌面 8 cm，仍雙臂接觸，維持 0.15 s | +10 |
| 搬運 | 箱子到右桌目標 11 cm 內，機器人到右站位 16 cm 內，仍雙臂接觸 0.15 s | +10 |
| 放下 | 完整箱體投影在右桌內、有向上支撐力、正立、低速，維持 0.25 s | +10 |
| 鬆手 | 持續由右桌支撐，雙前臂離箱 >6 cm、接觸力 <0.1 N，連續 1 s | +50 並成功結束 |

每階段另有 `8 × 有正負號的進展量`。走遠、抓點偏離會扣回進展，往返不能反覆刷分。
階段切換時重設進展基準，里程碑每回合只給一次。每秒另扣 0.40 分以減少拖延。
保留小權重的速度追蹤、步態，以及能量、姿態、腳滑、動作變化與關節限制懲罰。
移除走路任務的手臂回到固定站姿獎懲，避免妨礙抱箱；前臂與箱子的正常接觸不算不良碰撞。

跌倒、傾倒、掉箱、抓住後超過 0.4 s 失去雙臂接觸、搬運時箱底過低或機器人出界，失敗 -25。
超過 35 s 是 timeout 截斷；不另外當作跌倒扣 -25，仍承擔每秒時間成本。
Isaac Lab RewardManager 會乘控制時間；一次性獎勵／失敗扣分已做 dt 換算，不會因步長縮小而縮水。
上述都是初始權重，訓練後仍需依成功率和影片調整。

## 在這台 Windows 電腦上

已建立本專案的 `.venv`（Python 3.12），並安裝檢查、CPU PyTorch、MuJoCo 與回放套件。
不需要啟用虛擬環境，從專案資料夾執行：

```powershell
.\.venv\Scripts\python.exe tools\check_local.py
.\.venv\Scripts\python.exe tools\check_box_reach.py
.\.venv\Scripts\python.exe tools\preview_box_transfer.py
```

最後一個指令產生 `box_transfer_scene.png`。這是靜態場景示意，不是學會搬箱的影片。
雙擊 `VIEW_SCENE.cmd` 可開啟本機 3D 場景視窗；滑鼠左鍵拖曳旋轉、右鍵拖曳平移、滾輪縮放。
也可執行 `.\.venv\Scripts\python.exe tools\preview_box_transfer.py --live`。
此視窗維持初始姿勢，供檢查場景；不執行訓練或自動搬箱。
可達範圍檢查使用原廠 MJCF 的前向運動學，但限制採訓練用 URDF 的關節範圍及 0.90 soft limit；
它確認抓點能到達，不代表整段動作無碰撞、能維持平衡或已經學會抓取。
提交分支為 `student/zhaoyu-lu-HMD`，GitHub Fork 為 `yuki-beginner/LearningRL`。
助教請使用此分支上的最新 commit，先跑 smoke，再決定是否長訓練。
本機 Python 環境與 vendor 模型不會上傳；其他電腦請依下方步驟重新安裝。

重新建立本機環境時（需要 uv 與 Git）：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe ruff pytest gitman
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e robot_description -e LearningRLsim
$env:GITMAN_CACHE = Join-Path (Get-Location) '.gitman-cache'
.\.venv\Scripts\gitman.exe install
```

## 本機驗證紀錄

- 來源版本：`eb14d503fde3d5e5b0db109174b1372e5a82ffcf`。
- Ruff 格式與 lint：通過。
- `LearningRLlab/tests`：25 通過（含新增 16 個搬箱邏輯案例）。
- `robot_description/tests` ＋ `LearningRLsim/tests`：177 通過，2 略過（需要 Isaac Lab）。
- Windows `learningrl-sim --help`：可啟動；鍵盤解析有模擬輸入測試，未實際互動操控。
- MuJoCo 靜態場景已渲染並檢視；抓點可達範圍另外有數值檢查工具。
- 尚未在助教 GPU 上執行 Isaac Lab smoke、長訓練、物理抓取或錄影。

程式依據：[課程 README](https://github.com/mark-helwig/LearningRL/blob/main/README.md)、
[課程版本注意事項](https://github.com/mark-helwig/LearningRL/blob/main/CLAUDE.md)、
[原廠鎖定 URDF](https://github.com/Roboparty/rpo_description/blob/37aac9ca665e92731444a1618320078e7ba21569/urdf/rpo.urdf)。

## 給助教：訓練與錄影

沿用課程指定的 Isaac Lab `v3.0.0-beta2.patch1`、Newton/MuJoCo-Warp GPU 容器。
請勿直接換成網路上不同版本的 Isaac Lab 教學，API、四元數順序和陣列介面不同。
安裝順序仍是 robot_description → LearningRLlab → LearningRLsim（最後一個 `--no-deps`）。
以下假設專案依課程原本方式掛載在 `/workspace/isaaclab/source/`；在專案根目錄呼叫：

```bash
bash tools/train_box_transfer.sh smoke
bash tools/train_box_transfer.sh train
bash tools/train_box_transfer.sh video /absolute/path/to/model_20000.pt
```

先跑 smoke 確認場景能建立、感測器能取得指定物體的接觸力、reward 有限值。
若場景初始化、接觸篩選或物理步進出錯，先修正再長訓練。
video 指令使用 -PLAY 任務、單一環境、錄製 1750 控制步（35 s），並關閉 sim2sim 匯出。
日誌依課程 wrapper 規則寫入 `/workspace/isaaclab/logs/rsl_rl/learningrl_rpo_box_transfer/`。
本版使用 TensorBoard，不需要 Claude/API token 或 WandB 帳號。

完整的走路＋物體接觸操作比原本走路任務困難，20,000 iterations 是起始預算，不是保證成功。
若完全學不到抓取，先把 `reset_task` 的 `approach_distance` 改成 0.15 做桌邊練習；
穩定後再逐步增加至 0.4、0.7、1.0。每次改場景請重新記錄 commit 與訓練設定。
可繼續訓練同一個搬箱任務 checkpoint；原走路 policy 的觀察維度不同，不能直接當成可無縫 resume。
至少用多個 random seed 評估完整成功率、掉箱率與跌倒率，再挑選展示影片。

## CPU 回放的限制

原 `learningrl-sim ... --commands walk_baseline` 仍然只支援原先的走路等任務。
新增搬箱任務有自訂 TransferCommandCfg、task_observation 與物體接觸階段，尚未移植到該 ONNX 回放器。
請向助教取 MP4 或用上面 video 指令看這個任務。不能把它的 ONNX 塞進原回放器並期待搬箱。
本次也修正原回放器在 Windows 匯入 termios 會失敗的問題；Windows 鍵盤改用標準庫 msvcrt。

## 修改範圍與主要檔案

- `LearningRLlab/LearningRLlab/tasks/locomanipulation/rpo/box_env_cfg.py`：環境、獎懲、感測器、PLAY。
- 同資料夾 `layout.py`、`assets.py`：場景尺寸、五面板空箱與桌子。
- 同資料夾 `mdp/core.py`：階段、幾何與防止誤判的邏輯。
- 同資料夾 `mdp/commands.py`、`mdp/terms.py`：導航指令、觀察、reset、reward、termination。
- 同資料夾 `agents/rsl_rl_ppo_cfg.py`：PPO；關閉不適用此非對稱任務的鏡射資料增強。
- `LearningRLlab/tests/test_box_transfer.py`：CPU 任務邏輯測試。
- `tools/`：本機檢查、靜態預覽、助教訓練與錄影指令。

既有 `LearningRL-RPO-Walk-v0` 的設定未改成搬箱任務，仍可獨立使用。
