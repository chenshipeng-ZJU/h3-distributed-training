import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import pybind_xprofiler

# 使用案例1：手动插桩
with nvtx.range("[user define] example"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x, hidden=None):
            output, hidden = self.rnn(x, hidden)
            last_output = output[:, -1, :]  # (batch_size, hidden_size)
            result = self.fc(last_output)  # (batch_size, output_size)
            
            return result, hidden
    
    input_size = 10
    hidden_size = 20    
    output_size = 5    
    batch_size = 3     
    seq_len = 8        
    
    # 使用案例2：自动 NVTX 插桩，该范围内手动 NVTX 插桩会失效
    with torch.autograd.profiler.emit_nvtx():
        model = SimpleRNN(input_size, hidden_size, output_size).to(device)
        
        input_data = torch.randn(batch_size, seq_len, input_size).to(device)
        
        with torch.no_grad():
            output, hidden_state = model(input_data)
            
        print(f"输出形状: {output.shape}")
        print(f"输出设备: {output.device}")
        print(f"隐藏状态形状: {hidden_state.shape}")
        print(f"隐藏状态设备: {hidden_state.device}")
    
        # 使用案例3：开启 profiler，适用于 daemon 模式
        pybind_xprofiler.cuda_profiler_start()
    
        output_cpu = output.cpu()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        target = torch.randn(batch_size, output_size).to(device)
        
        optimizer.zero_grad()
        output, _ = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
        # 使用案例3：关闭 profiler，适用于 daemon 模式
        pybind_xprofiler.cuda_profiler_stop()
    
