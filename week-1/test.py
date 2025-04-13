import torch

# 預測的前2名類別
top2_pred = torch.tensor([
    [0, 1],
    [2, 3],
    [4, 5]
]) 

# 實際類別
y = torch.tensor([5, 3, 1])

# torch.isin（會將y比對整體 top2_pred，而非逐項比對top2_pred細項）
correct_top2_isin = torch.isin(y, top2_pred).sum().item()

# 逐筆比對 top-2 是否命中
match_top2 = (top2_pred[:, 0] == y) | (top2_pred[:, 1] == y)
correct_top2 = match_top2.sum().item()

print("正確值 y：\n", y)
print("Top-2 預測：\n", top2_pred)
print("原作法用torch.isin 整體比對（結果高估）,", "correct:",correct_top2_isin, ",",torch.isin(y, top2_pred))
print("調整為逐項比對,","correct:", correct_top2,",", match_top2)