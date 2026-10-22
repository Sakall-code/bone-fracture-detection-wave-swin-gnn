# Fix Bugs in `Wave-Swin-GNN30.ipynb`

Apply the following 3 fixes to the file `Wave-Swin-GNN30.ipynb` located in the current directory. **Do not modify any other files.**

---

## Fix 1 — Cell 6 (`CELL 6: COMPREHENSIVE EVALUATION`)

**Bug:** `KeyError: 'source'` on the line:
```python
test_sources_list = [d['source'] for d in test_data]
```
The data entries only have `'path'` and `'label'` keys — there is no `'source'` key.

**Fix:**
1. Change the initialization line from:
   ```python
   test_labels_list, test_scores_list, test_sources_list = [], [], []
   ```
   to:
   ```python
   test_labels_list, test_scores_list = [], []
   ```
2. Delete the comment `# Track source for each test sample` and the line `test_sources_list = [d['source'] for d in test_data]` entirely.
3. Clear the cell's `"outputs"` array (set it to `[]`) to remove the stale error traceback.

---

## Fix 2 — Cell 7 (`CELL 7: EVALUATION PLOTS`)

**Bug:** Subplot 6 ("Per-Dataset Performance") references the now-removed `test_sources_list` variable and tries to split results by dataset keys `'grazpedwri'` and `'multiregion'`, which don't exist in the data.

**Fix:** Replace the entire **subplot 6 block** (from `# 6. Per-Dataset Performance` up to but NOT including `plt.tight_layout()`) with this simpler metrics summary chart:

```python
# 6. Summary Metrics Bar Chart
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AP', 'AUROC']
metric_values = [acc, prec, rec, f1, ap, auroc]
colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'goldenrod', 'tomato']
x_pos = np.arange(len(metric_names))
axes[1, 2].bar(x_pos, metric_values, color=colors)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(metric_names, fontsize=10, rotation=30, ha='right')
axes[1, 2].set_ylabel('Score', fontsize=12)
axes[1, 2].set_title('Test Set Metrics Summary', fontsize=14)
axes[1, 2].set_ylim(0, 1.05)
for i, v in enumerate(metric_values):
    axes[1, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
```

---

## Fix 3 — Cell 10 (`CELL 10: MODEL SUMMARY & EXPORT`)

**Bug:** Hardcoded batch size `8` instead of the actual `BATCH_SIZE` variable (which is `16`).

**Fix:** In the f-string, change:
```python
f"  Effective batch size: {8 * ACCUMULATION_STEPS}"
```
to:
```python
f"  Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}"
```
