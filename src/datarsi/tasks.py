from dataclasses import dataclass
import re
from .storage import read_jsonl


@dataclass
class Task:
    task_id: str
    split: str
    problem: str
    image: str
    repo_dir: str
    verify_command: str
    base_commit: str | None=None
    gold_patch: str | None=None
    test_patch: str | None=None


def load_tasks(path,split):
    tasks=[]
    seen=set()
    for row in read_jsonl(path):
        task=Task(**row)
        if task.split!=split:
            raise ValueError(f'{task.task_id}: expected {split} split')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',task.task_id) or task.task_id in seen:
            raise ValueError('Invalid or duplicate task_id')
        if not task.repo_dir.startswith('/') or not task.verify_command.strip():
            raise ValueError('Task needs an absolute repo_dir and verification command')
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError('Task set is empty')
    return tasks


def check_disjoint(train,dev):
    overlap={t.task_id for t in train}&{t.task_id for t in dev}
    if overlap:
        raise ValueError(f'Train/dev task overlap: {sorted(overlap)}')
