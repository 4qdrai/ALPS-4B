import json

def main():
    log_path = "results/two_rooms/training_log.json"
    with open(log_path, "r") as f:
        log = json.load(f)
        
    print(f"Total epochs in training log: {len(log)}")
    print("-" * 80)
    print(f"{'Epoch':<6} | {'Total Loss':<10} | {'Pred Loss':<10} | {'SIGReg':<10} | {'VQ Loss':<10} | {'MoE Loss':<10} | {'Sys2 Act':<8}")
    print("-" * 80)
    
    epochs_to_print = [0, 1, 2, 3, 4, 9, 19, 29, 39, 49]
    for i in epochs_to_print:
        if i < len(log):
            e = log[i]
            print(f"{e['epoch']:<6d} | {e['total_loss']:<10.4f} | {e['pred_loss_op']:<10.6f} | {e['sigreg_loss']:<10.6f} | {e['vq_loss']:<10.6f} | {e['moe_loss']:<10.6f} | {e['system2_activation_count']:<8d}")
    print("-" * 80)

if __name__ == "__main__":
    main()
