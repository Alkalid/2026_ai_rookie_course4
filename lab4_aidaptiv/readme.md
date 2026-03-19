# lab4_aidaptiv 執行流程

這個 Lab 請先啟動 Docker 服務，再進入容器執行訓練指令。  
**順序一定是：先 `docker compose up`，再跑下面的 command。**

## 1) 前置準備

- 已安裝 Docker Desktop（或等效 Docker 環境）
- Docker 引擎已啟動（Docker Desktop 開著）
- 目前目錄在 `lab4_aidaptiv`

## 2) 先啟動容器（必要）

在 `lab4_aidaptiv` 目錄執行：

```bash
docker compose up -d
```

說明：
- `-d` 代表背景執行，容器會持續在背景跑
- 若你想看啟動 log，可先用 `docker compose up`（不加 `-d`）

可用下列指令確認容器有正常起來：

```bash
docker compose ps
```

## 3) 進入容器

啟動成功後，再進入容器執行 Lab 指令。  
依目前 `docker-compose.yaml`，服務名稱是 `aidaptiv_fine_tune`，可用：

```bash
docker compose exec aidaptiv_fine_tune bash
```

> 若你後續有改過服務名稱，請改成你在 `docker compose ps` 看到的名稱。

## 4) 在容器內執行指令

進入容器後執行：

```bash
cp /workspace/env_config.yaml /home/root/aiDAPTIV2/commands/env_config/
cp /workspace/exp_config.yaml /home/root/aiDAPTIV2/commands/exp_config/
cp /workspace/QA_dataset_config.yaml /home/root/aiDAPTIV2/commands/dataset_config/text-generation/

cd /home/root/aiDAPTIV2/commands/
phisonai2 --env_config ./env_config/env_config.yaml --exp_config ./exp_config/exp_config.yaml
```

## 5) 常見問題

- `Cannot connect to the Docker daemon`  
  代表 Docker 還沒啟動，請先開 Docker Desktop。
- `service ... is not running`  
  代表你還沒 `docker compose up -d`，或容器啟動失敗，先看 `docker compose ps` / `docker compose logs`。
- `docker compose exec` 失敗  
  通常是服務名稱打錯，請先用 `docker compose ps` 確認名稱。