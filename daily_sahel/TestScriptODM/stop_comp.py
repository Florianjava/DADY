from pyodm import Node
node = Node("localhost", 3000)

# Liste les tâches
tasks = node.list_tasks()
for task in tasks:
    print(f"{task.uuid} - {task.info().status.name}")

# Pour annuler une tâche en cours (ex: la plus récente)
task = tasks[-1]  # ou filtre selon UUID
task.cancel()
print("✅ Tâche annulée proprement.")