import torch 
from torch import amp 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.nn.functional as distnn
import argparse
import os  
from scipy.stats import spearmanr, pearsonr
import json 
import copy 

from patch_sampling.Patch_Sampler import SaliencyPatchSampler 
from model.IQA_Model import IQA_Model
from patch_sampling.Patch_Sampler import SaliencyPatchSampler
from data_retrieval.master_loader import MasterLoader 
from losses.metric_loss_uncertainty import MetricLossUncertainty
from losses.metric_loss import MetricLoss
from losses.metric_plcc import Metric_PLCC
from log.logging import Logging
from utils.utils import * 


def setup(rank, world_size, local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{local_rank}"))


def trainOneEpoch(epoch: int,
                  master_loader: MasterLoader,
                  model: torch.nn.Module,
                  model_ema: torch.nn.Module,
                  patch_sampler: torch.nn.Module,
                  patch_sampler_ema: torch.nn.Module,
                  lossFn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  global_rank: int,
                  args: dict): 
    
    scaler = amp.GradScaler(device)  # or dtype-specific below
    autocast_dtype = torch.bfloat16 
    
    master_loader.setEpoch(epoch) 
    loaders = master_loader.TrainLoaders()  
    loader_iters = [iter(loader) for loader in loaders] 
    
    if global_rank==0: 
        print(f"\nEpoch: {epoch}") 

    model.module.train() 
    patch_sampler.module.train()  
    for current_iteration in range(master_loader.epochLength()): 
        optimizer.zero_grad() 
        predictions = [] 
        gts = []  
        shapes = []  
        for idx, loader_iter in enumerate(loader_iters): 
            batch_l = next(loader_iter, None) # list[dict]
            if batch_l is None: 
                loader_iters[idx] = iter(loaders[idx])    
                batch_l = next(loader_iters[idx], None) # list[dict] 

            for i, sample in enumerate(batch_l): 
                # Sample patches and get the weights.  
                image, weights, scale_counts = patch_sampler(sample["image"].to(device, non_blocking=True),
                                                                num_patches=args["train_num_patches"],
                                                                stride_ratio=None,
                                                                random=True) 
                batch_l[i]["image"] = image
                batch_l[i]["weights"] = weights   
                batch_l[i]["scale_counts"] = scale_counts 

            batch = {key: torch.stack( [sample[key] for sample in batch_l] ) for key in batch_l[0].keys()} 
            
            # IQA model forward pass. 
            with amp.autocast(device_type="cuda", dtype=autocast_dtype):
                prediction = model(batch["image"], batch["weights"], batch["scale_counts"][0, ...]) 

            # Append predictions and gts, along with the batch size per dataset. 
            predictions.append(prediction) # (B) 
            gts.append(batch["quality"].to(device)) # (B) 
            shapes.append(prediction.shape[0]) 

        indices = [0] 
        for shape in shapes: 
            indices.append(indices[-1] + shape)  
        
        predictions = torch.cat(predictions, dim=0)
        gts = torch.cat(gts, dim=0) 

        predictions_gathered = torch.stack(distnn.all_gather(predictions), dim=0) # (WORLD_SIZE, B1+B2...)  
        gts_gathered = torch.stack(distnn.all_gather(gts), dim=0) # (WORLD_SIZE, B1+B2...)    

        predictions_reorged = [] 
        gts_reorged = []  
        for i in range(len(indices)-1): 
            predictions_reorged.append(predictions_gathered[:, indices[i]:indices[i+1]].reshape(-1)) 
            gts_reorged.append(gts_gathered[:, indices[i]:indices[i+1]].reshape(-1)) 

        loss, meta = lossFn(predictions_reorged, gts_reorged) # predictions and gts are lists.   
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if model_ema:
            update_ema(model.module, model_ema, decay=0.999)
        if patch_sampler_ema:
            update_ema(patch_sampler.module, patch_sampler_ema, decay=0.999)

        if global_rank == 0 and (current_iteration + 1) % args["print_every"] == 0:
            print(f"{current_iteration + 1}/{master_loader.epochLength()} | ", end="")
            print(" | ".join(f"{key.upper()}: {meta[key]:.6f}" for key in meta))


@torch.no_grad()
def evaluate(Logger: Logging,
             epoch: int,
             DataMaster: MasterLoader,
             dataset_names: list, 
             model: torch.nn.Module, 
             model_ema: torch.nn.Module, 
             patch_sampler: torch.nn.Module, 
             patch_sampler_ema: torch.nn.Module, 
             optimizer: torch.optim.Optimizer,
             device: torch.device,
             global_rank: int,
             args: dict): 
    if global_rank==0: 
        print(f"\nEvaluation Epoch: {epoch}") 

    loaders = DataMaster.TestLoaders()
    
    if global_rank == 0: 
        Logger.append(key="lr", value=optimizer.param_groups[0]['lr']) 
        Logger.append(key="epochs", value=epoch) 
    
    model.module.eval() 
    patch_sampler.module.eval()  
    models = {"primary": model, "ema": model_ema}
    patch_samplers = {"primary": patch_sampler, "ema": patch_sampler_ema} 

    for (model_name, test_model) in models.items(): 
        test_patch_sampler = patch_samplers[model_name] 
        plcc_performances = []
        srcc_performances = [] 
        for (name, loader) in zip(dataset_names, loaders): 
            quality_predictions = [] 
            quality_gts = [] 

            for i, batch_l in enumerate(loader): 
                for i, sample in enumerate(batch_l): 
                    # Sample patches and get the weights.  
                    image, weights, scale_counts = test_patch_sampler(sample["image"].to(device, non_blocking=True),
                                                                      num_patches=args["train_num_patches"],
                                                                      stride_ratio=args["patch_stride"],
                                                                      random=False) 
                    batch_l[i]["image"] = image
                    batch_l[i]["weights"] = weights   
                    batch_l[i]["scale_counts"] = scale_counts 

                batch = {key: torch.stack( [sample[key] for sample in batch_l] ) for key in batch_l[0].keys()} 
            
                # IQA  model forward pass. 
                prediction = test_model(batch["image"], batch["weights"], batch["scale_counts"][0, ...]).detach()  
                quality_predictions.append(prediction)  
                quality_gts.append(batch["quality"].to(device))
            
            quality_predictions = torch.cat(quality_predictions, dim=0) 
            quality_gts = torch.cat(quality_gts, dim=0) 
            
            gathered_predictions = [torch.zeros_like(quality_predictions) for _ in range(args['world_size'])]
            gathered_gts = [torch.zeros_like(quality_gts) for _ in range(args['world_size'])]
            dist.barrier() 
            dist.all_gather(gathered_predictions, quality_predictions)
            dist.all_gather(gathered_gts, quality_gts)
            
            gathered_predictions = torch.cat(gathered_predictions, dim=0).reshape(-1).cpu().numpy()
            gathered_gts = torch.cat(gathered_gts).reshape(-1).cpu().numpy() 
            
            srcc, _ = spearmanr(gathered_predictions, gathered_gts) 
            plcc, _ = pearsonr(gathered_predictions, gathered_gts)  
            
            srcc_performances.append(srcc) 
            plcc_performances.append(plcc) 
            
            if global_rank == 0: 
                Logger.append(key=f"{model_name}_test_{name}_srcc", value=srcc) 
                Logger.append(key=f"{model_name}_test_{name}_plcc", value=plcc) 
                
        srcc_performance = sum(srcc_performances) / len(srcc_performances) 
        plcc_performance = sum(plcc_performances) / len(plcc_performances) 
        if global_rank == 0: 
            Logger.append(key=f"{model_name}_test_srcc_avg", value=srcc_performance) 
            Logger.append(key=f"{model_name}_test_plcc_avg", value=plcc_performance) 


@torch.no_grad()
def update_ema(model, model_ema, decay=0.999):
    msd = model.state_dict()
    grad_params = {k for k, v in model.named_parameters() if v.requires_grad}  # names of trainable params
    
    for k, ema_v in model_ema.state_dict().items():
        if k in msd and any(k.startswith(gp) for gp in grad_params):  # only update trainable params
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

def saveModel(epoch, model, model_ema, patch_sampler, patch_sampler_ema, optimizer, scheduler, args): 
    state = {"model": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
             "model_ema": model_ema.module.state_dict() if isinstance(model_ema, torch.nn.parallel.DistributedDataParallel) else model_ema.state_dict(),
             "patch_sampler": patch_sampler.module.state_dict() if isinstance(patch_sampler, torch.nn.parallel.DistributedDataParallel) else patch_sampler.state_dict(),
             "patch_sampler_ema": patch_sampler_ema.module.state_dict() if isinstance(patch_sampler_ema, torch.nn.parallel.DistributedDataParallel) else patch_sampler_ema.state_dict()}
    torch.save(state, os.path.join(args["save_dir"], f"checkpoint_{epoch}.pth")) 


def train(args):
    # Perform the distributed training initialization.
    setup(
        rank=args["global_rank"],
        world_size=args['world_size'],
        local_rank=args["local_rank"]
    )
    global_rank = args["global_rank"]
    local_rank = args["local_rank"]

    print(f"[Global Rank {global_rank}] Initialized on GPU {local_rank}", flush=True)

    # Setup. 
    device = torch.device(f"cuda:{local_rank}")  

    if global_rank == 0: 
        os.makedirs(args["save_dir"], exist_ok=True)
    
    dataset_names = args["dataset_names"]
    samples = args["samples"] 

    Logger = Logging(plot_dir=args["plot_dir"]) 
    
    patch_sampler = SaliencyPatchSampler(patch_size = args["patch_size"],
                                         num_scales = args["num_scales"], 
                                         min_short_edge = args["min_short_edge"],
                                         device=device).to(device)  

    DataMaster = MasterLoader(dataset_names,
                              samples, 
                              device=device, 
                              options=args) 

    model = IQA_Model(device=device, options=args).to(device) 

    lossFn = MetricLossUncertainty().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(lossFn.parameters()) + list(patch_sampler.parameters()),
        lr=args["lr"],
        weight_decay=args["weight_decay"]
        ) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["scheduler_T_max"])
    
    patch_sampler = DDP(patch_sampler, device_ids=[local_rank], find_unused_parameters=True)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    lossFn = DDP(lossFn, device_ids=[local_rank], find_unused_parameters=True)  
    dist.barrier() 
    
    # Start with pooling and patch sampling only.  
    model.module.freeze_model() 
    model.module.unfreeze_pooling()  
    patch_sampler.module.unfreeze()  
    patch_sampler.module.freeze_ve() 
    
    for i in range(args["init_epochs"]):
        print(f"Initial Stage Epoch: -1 {i}/{args['init_epochs']}") 
        trainOneEpoch(-1, DataMaster, model, None, patch_sampler, None, lossFn, optimizer, device, global_rank, args)

    # Initialize the EMAs after the initial training.  
    model_ema = copy.deepcopy(model.module).to(device) 
    for p in model_ema.parameters():
        p.requires_grad_(False)
    patch_sampler_ema = copy.deepcopy(patch_sampler.module).to(device) 
    for p in patch_sampler_ema.parameters():
        p.requires_grad_(False)
     
    model_ema.eval() 
    patch_sampler_ema.eval()  

    # First evaluation. 
    evaluate(Logger, -1, DataMaster, dataset_names, model, model_ema, patch_sampler, patch_sampler_ema, optimizer, device, global_rank, args)  
    if global_rank == 0:
        Logger.plot(x_axis_key="epochs")  
        saveModel(-1, model, model_ema, patch_sampler, patch_sampler_ema, optimizer, scheduler, args) 
    

    # End-to-end training
    model.module.unfreeze_vision_encoder() 
    model.module.unfreeze_pooling()  
    patch_sampler.module.unfreeze()
    for epoch in range(args["epochs"]):     
        trainOneEpoch(epoch, DataMaster, model, model_ema, patch_sampler, patch_sampler_ema, lossFn, optimizer, device, global_rank, args)
        evaluate(Logger, epoch, DataMaster, dataset_names, model, model_ema, patch_sampler, patch_sampler_ema, optimizer, device, global_rank, args)  

        if global_rank == 0:
            Logger.plot(x_axis_key="epochs")  
            saveModel(epoch, model, model_ema, patch_sampler, patch_sampler_ema, optimizer, scheduler, args) 
        
        scheduler.step() 
        if optimizer.state_dict()['param_groups'][0]['lr'] == 0: 
            scheduler.step() 


    dist.barrier()
    dist.destroy_process_group()


def main(**kwargs): 
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    local_rank = int(os.environ['LOCAL_RANK']) 
    global_rank = int(os.environ["RANK"])      
    world_size = int(os.environ["WORLD_SIZE"]) 
    
    gpus_per_node = torch.cuda.device_count()

    args = {"master_addr": master_addr,
            "master_port": master_port,
            "local_rank": local_rank,
            "global_rank": global_rank,
            "gpus_per_node": gpus_per_node,
            "world_size": world_size,
            **kwargs}

    train(args)
    

if __name__ == "__main__": 
    import sys

    # Get version as a tuple
    version_info = sys.version_info
    print(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")

    parser = argparse.ArgumentParser(description="VLM IQA training code.")
    parser.add_argument("--run_dir", type=str, default=os.path.join(".", "baseline"), help="Everything related to run.")
    parser.add_argument("--config", type=str, default=os.path.join("options", "default_options.json"), help="Path to config.json")
    args = parser.parse_args() 

    config = {}
    with open(args.config, "r") as f:
        config = json.load(f)
    
    merged_config = {} 
    for subdict in config.values(): 
        merged_config.update(subdict) 
    
    merged_config["save_dir"] = os.path.join(args.run_dir, merged_config["save_dir"])
    merged_config["plot_dir"] = os.path.join(args.run_dir, merged_config["plot_dir"])
    
    main(**merged_config) 
