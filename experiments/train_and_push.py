import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from dotenv import load_dotenv

import logging
import warnings

warnings.filterwarnings("ignore")
for logger_name in ["mlflow", "mlflow.sklearn", "mlflow.models.model"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Завантажуємо змінні з .env файлу
load_dotenv()

# Налаштування з оточення (з фолбеком на localhost для port-forward)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091") # Зверніть увагу: без http://

# Налаштовуємо MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_name = "Iris Classification Automated"
mlflow.set_experiment(experiment_name)

# Налаштовуємо Prometheus метрики
registry = CollectorRegistry()
g_accuracy = Gauge('mlflow_accuracy', 'Accuracy of the model', ['run_id'], registry=registry)
g_loss = Gauge('mlflow_loss', 'Log Loss of the model', ['run_id'], registry=registry)

# Завантажуємо датасет Iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Сітка гіперпараметрів для кількох запусків (Grid Search)
param_grid = [
    {"learning_rate": 0.01, "epochs": 50},
    {"learning_rate": 0.1, "epochs": 100},
    {"learning_rate": 1.0, "epochs": 200},
    {"learning_rate": 0.5, "epochs": 150}
]

print("🚀 Починаємо серію експериментів...")

# Цикл тренування
for params in param_grid:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        print(f"⏳ Запуск тренування {run_id} з параметрами: {params}")
        mlflow.log_params(params)

        # C - це обернений learning_rate в LogisticRegression
        model = LogisticRegression(C=params["learning_rate"], max_iter=params["epochs"], solver='lbfgs')
        model.fit(X_train, y_train)

        # Прогнозування та метрики
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        # Логуємо метрики та модель в MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("loss", loss)
        mlflow.sklearn.log_model(model, "model")

        print(f"   ✅ Завершено. Accuracy: {acc:.4f} | Loss: {loss:.4f}")

        # Відправляємо метрики в Prometheus PushGateway
        g_accuracy.labels(run_id=run_id).set(acc)
        g_loss.labels(run_id=run_id).set(loss)
        try:
            push_to_gateway(PUSHGATEWAY_URL, job='mlflow_experiments', registry=registry)
            print("   📈 Метрики успішно відправлено в PushGateway")
        except Exception as e:
            print(f"   ⚠️ Помилка відправки в PushGateway: {e}")

print("\n🔍 Шукаємо найкращу модель...")
client = MlflowClient()
experiment = mlflow.get_experiment_by_name(experiment_name)

# Використовуємо Pandas Dataframe з API MLflow для пошуку найкращого результату
best_run = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC", "metrics.loss ASC"],
    max_results=1
).iloc[0]

best_run_id = best_run.run_id
best_acc = best_run["metrics.accuracy"]
print(f"🏆 Найкращий Run ID: {best_run_id} з accuracy {best_acc:.4f}")

print("\n💾 Завантажуємо найкращу модель локально...")
os.makedirs("../best_model", exist_ok=True) # Створюємо папку best_model в корені проєкту

# Завантажуємо артефакт
local_path = mlflow.artifacts.download_artifacts(
    run_id=best_run_id, 
    artifact_path="model", 
    dst_path="../best_model/"
)
print(f"🎉 Модель збережено у папці: {os.path.abspath(local_path)}")