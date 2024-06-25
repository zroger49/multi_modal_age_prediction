from ..params import *
import torch.nn as nn
import random
import torch.nn.functional as F
import itertools

def generate_nodes(n_layers, unit_options=[64, 128, 256, 512, 1024]):
    import itertools

    values = unit_options
    N = n_layers

    combinations = list(itertools.product(values, repeat=N))

    lt_n_nodes = list(random.choice(combinations))
    lt_n_nodes.sort(reverse=True)

    return lt_n_nodes

def define_model_static(num_layers, n_units, p_dropout, in_features):
    layers = []

    for i in range(num_layers):
        layers.append(nn.Linear(in_features, n_units[i]))
        layers.append(nn.ReLU())
        #layers.append(nn.BatchNorm1d(n_units[i]))
        layers.append(nn.Dropout(p_dropout[i]))

        in_features = n_units[i]

    #layers.append(nn.Linear(in_features, 1))

    return nn.Sequential(*layers), in_features

def define_branch(trial, in_features, branch_name=""):
    num_layers = trial.suggest_int(f"{branch_name}_n_layers", n_layers["min"], n_layers["max"])
    layers = []

    unit_options = [2**i for i in range(n_nodes["min"], n_nodes["max"]+1)]
    if n_nodes["keep_cte"]:
        unit_options.append(unit_options[0])

    lt_nodes = generate_nodes(num_layers, unit_options)
    for i in range(num_layers):
        trial.set_user_attr(f"{branch_name}_n_units_l{i}", lt_nodes[i])
        layers.append(nn.Linear(in_features, lt_nodes[i]))
        #layers.append(nn.BatchNorm1d(lt_nodes[i]))  # Add BN after each Linear layer
        layers.append(nn.ReLU())
        p = trial.suggest_float(f"{branch_name}_dropout_l{i}", dropout_rate["min"], dropout_rate["max"])
        layers.append(nn.Dropout(p))
        in_features = lt_nodes[i]

    # layers.append(nn.Linear(in_features, 64))
    # in_features = 64
    return nn.Sequential(*layers), in_features

class MultiNetDefault(nn.Module):
    def __init__(self, trial, in_feat__hist, in_feat__meth):
        super(MultiNetDefault, self).__init__()
        
        self.model__hist, out_feat__hist = define_branch(
            trial, 
            in_features=in_feat__hist, 
            branch_name="hist"
            )
        
        self.model__meth, out_feat__meth = define_branch(
            trial, 
            in_features=in_feat__meth, 
            branch_name="meth"
            )

        self.features__comb = out_feat__hist + out_feat__meth
        
        self.linear = nn.Linear(self.features__comb, 1)

    def forward(self, features__hist, features__meth):
        output__hist = self.model__hist(features__hist)
        output__meth = self.model__meth(features__meth)


        output__comb = torch.cat((output__hist, output__meth), dim=1)

        pred__age = self.linear(output__comb)

        return pred__age

class MultiNetDynamicOld(nn.Module):
    def __init__(self, trial, modalities_features):
        super(MultiNetDynamic, self).__init__()
        
        self.branches = nn.ModuleDict()
        total_out_features = 0
        
        # Dynamically create branches for each modality
        for modality, in_features in modalities_features.items():
            branch, out_features = define_branch(
                trial, 
                in_features=in_features, 
                branch_name=modality
            )
            self.branches[modality] = branch
            total_out_features += out_features
        
        # Combine features from all modalities
        self.features_comb = int(128*len(modalities_features))#total_out_features
        self.linear = nn.Linear(self.features_comb, 1)

    def forward(self, **features):
        outputs = []

        # Process each modality
        for modality, branch in self.branches.items():
            if modality in features:
                modality_features = features[modality]
                output = branch(modality_features)
                outputs.append(output)
        
        # Concatenate outputs from all modalities
        output_comb = torch.cat(outputs, dim=1)
        pred_age = self.linear(output_comb)

        return pred_age

class MultiNetDynamic(nn.Module):
    def __init__(self, trial, modalities_features):
        super(MultiNetDynamic, self).__init__()
        
        self.branches = nn.ModuleDict()
        total_out_features = 0
        
        # Dynamically create branches for each modality
        for modality, in_features in modalities_features.items():
            branch, out_features = define_branch(
                trial, 
                in_features=in_features, 
                branch_name=modality
            )
            self.branches[modality] = branch
            total_out_features += out_features
        
        self.features_comb = total_out_features

        self.fc1 = nn.Linear(self.features_comb, 1)
        # self.dropout1 = nn.Dropout(0.2)

        # self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(0.2)

        # self.fc3 = nn.Linear(64, 1)
        
    def forward(self, **features):
        outputs = []

        # Process each modality
        for modality, branch in self.branches.items():
            if modality in features:
                modality_features = features[modality]
                output = branch(modality_features)
                outputs.append(output)
        
        # Concatenate outputs from all modalities
        output_comb = torch.cat(outputs, dim=1)
        
        # Pass through additional layers
        # x = F.relu(self.fc1(output_comb))
        # x = self.dropout1(x)

        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)

        pred_age = self.fc1(output_comb)

        return pred_age

class MultiNetDynamicV2(nn.Module):
    def __init__(self, trial, modalities_features):
        super(MultiNetDynamic, self).__init__()
        
        self.branches = nn.ModuleDict()
        total_out_features = 0
        
        # Dynamically create branches for each modality
        for modality, in_features in modalities_features.items():
            branch, out_features = define_branch(
                trial, 
                in_features=in_features, 
                branch_name=modality
            )
            self.branches[modality] = branch
            total_out_features += out_features
        
        self.features_comb = total_out_features

        self.fc1 = nn.Linear(self.features_comb, 128)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, **features):
        outputs = []

        # Process each modality
        for modality, branch in self.branches.items():
            if modality in features:
                modality_features = features[modality]
                output = branch(modality_features)
                outputs.append(output)
        
        # Concatenate outputs from all modalities
        output_comb = torch.cat(outputs, dim=1)
        
        # Pass through additional layers
        x = F.gelu(self.fc1(output_comb))
        x = self.dropout1(x)

        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)

        pred_age = self.fc3(x)
        
        return pred_age

class MultiNetDynamicBilin(nn.Module):
    def __init__(self, trial, modalities_features):
        super(MultiNetDynamic, self).__init__()
        
        self.branches = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()
        self.bilinear_layers = nn.ModuleDict()
        branch_out_features = {}

        for modality, in_features in modalities_features.items():
            branch, out_features = define_branch(
                trial, 
                in_features=in_features, 
                branch_name=modality
            )
            self.branches[modality] = branch
            branch_out_features[modality] = out_features
            self.layer_norms[modality] = nn.LayerNorm(out_features)

        # Initialize bilinear layers for each pair of branches using their output features
        for (modality1, out_features1), (modality2, out_features2) in itertools.combinations(branch_out_features.items(), 2):
            key = f"{modality1}_{modality2}"
            self.bilinear_layers[key] = nn.Bilinear(out_features1, out_features2, 64) # Example output size
        
        # The number of bilinear combinations generated
        num_bilinear_combinations = len(self.bilinear_layers)
        
        # Assuming each bilinear layer produces an output of size 64, adjust this as needed
        self.fc1 = nn.Linear(num_bilinear_combinations * 64, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, **features):
        branch_outputs = {}
        for modality, branch in self.branches.items():
            if modality in features:
                out = branch(features[modality])
                # Apply layer normalization to each branch's output
                norm_out = self.layer_norms[modality](out)
                branch_outputs[modality] = norm_out
        
        bin_outputs = []
        # Process outputs through their corresponding bilinear layers
        for key, bilinear in self.bilinear_layers.items():
            modality1, modality2 = key.split('_')
            if modality1 in branch_outputs and modality2 in branch_outputs:
                output1 = branch_outputs[modality1]
                output2 = branch_outputs[modality2]
                bilin_out = bilinear(output1, output2)
                bin_outputs.append(bilin_out)

        # Concatenate bilinear layer outputs
        if bin_outputs:
            output_comb = torch.cat(bin_outputs, dim=1)
        else:
            output_comb = torch.zeros((features[next(iter(features))].shape[0], 64 * len(self.bilinear_layers)), device=next(iter(branch_outputs.values())).device)

        x = F.gelu(self.fc1(output_comb))
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        pred = self.fc3(x)
        
        return pred

class MultiNetDynamicAttention(nn.Module):
    def __init__(self, trial, modalities_features):
        super(MultiNetDynamicAttention, self).__init__()

        self.branches = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()

        total_out_features = 0

        for modality, in_features in modalities_features.items():
            # branch, out_features = define_branch(
            #     trial, 
            #     in_features=in_features, 
            #     branch_name=modality
            # )

            branch, out_features = define_model_static(
                num_layers=3,
                n_units=[512, 256, 128],
                p_dropout=[0.1,0.1,0.1],
                in_features=in_features, 
            )

            self.branches[modality] = branch
            self.layer_norms[modality] = nn.LayerNorm(out_features)

            attention = nn.MultiheadAttention(
                embed_dim=out_features, 
                num_heads=4,
                batch_first=True,
                device=DEVICE,
                dropout=0.5
                )
            
            self.attentions[modality] = attention

            total_out_features += out_features

        self.features_comb = total_out_features

        self.fc1 = nn.Linear(self.features_comb, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, **features):
        outputs = []

        for modality, branch in self.branches.items():
            if modality in features:
                modality_features = features[modality]
                
                output = branch(modality_features).unsqueeze(0)
                output = self.layer_norms[modality](output)
                attention_output, _ = self.attentions[modality](
                    output, 
                    output, 
                    output
                    )
                
                outputs.append(attention_output.squeeze(0))

        output_comb = torch.cat(outputs, dim=1)

        x = F.gelu(self.fc1(output_comb))
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        pred_age = self.fc3(x)

        return pred_age

class MultiNetAttention(nn.Module):
    def __init__(self, trial, in_feat__hist, in_feat__meth):
        super(MultiNetAttention, self).__init__()
        
        n_heads_meth = trial.suggest_categorical("meth_n_heads", [1, 2, 4, 8])
        n_heads_hist = trial.suggest_categorical("hist_n_heads", [1, 2, 4, 8])

        assert in_feat__hist % n_heads_hist == 0, f"in_feat__hist ({in_feat__hist}) must be divisible by n_heads_hist ({n_heads_hist})"
        assert in_feat__meth % n_heads_meth == 0, f"in_feat__meth ({in_feat__meth}) must be divisible by n_heads_meth ({n_heads_meth})"

        p_drop_meth = trial.suggest_float(f"p_drop_meth", 0, 0.2)
        p_drop_hist = trial.suggest_float(f"p_drop_hist", 0, 0.2)
        
        self.attention_hist = torch.nn.MultiheadAttention(
            embed_dim=in_feat__hist, 
            num_heads=n_heads_hist, 
            dropout=p_drop_hist, 
            bias=True,
            add_bias_kv=False, 
            add_zero_attn=False, 
            kdim=None, 
            vdim=None,
            # Althoug the first dim is the bs, I wanna consider it as the seq_lengh
            # to force attention to find correlations between patients
            batch_first=False,
            device=DEVICE, 
            dtype=None
            )
        
        self.attention_meth = torch.nn.MultiheadAttention(
            embed_dim=in_feat__meth, 
            num_heads=n_heads_meth, 
            dropout=p_drop_meth, 
            bias=True, 
            add_bias_kv=False, 
            add_zero_attn=False, 
            kdim=None, 
            vdim=None,
            batch_first=False, 
            device=DEVICE, 
            dtype=None
            )

        self.features__comb = in_feat__hist + in_feat__meth

        self.reducer01 = nn.Linear(self.features__comb, 256)
        self.prediction = nn.Linear(256, 1)
        
    def forward(self, features__hist, features__meth):
        
        output__hist_att = self.attention_hist(features__hist, features__hist, features__hist)[0]
        output__meth_att = self.attention_meth(features__meth, features__meth, features__meth)[0]
        #print(output__hist_att.shape, output__meth_att.shape)

        output__comb = torch.cat((output__hist_att, output__meth_att), dim=1)
        #print(output__comb.shape)
        
        output__reduced = self.reducer01(output__comb)
        output__reduced = nn.Dropout(0.1)(nn.ReLU()(output__reduced))
        pred__age = self.prediction(output__reduced)
        return pred__age

