import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward

def FFT_for_Period(x, k=2):

    B, T, C = x.shape

    #  计算自相关
    x_centered = x - x.mean(dim=1, keepdim=True)  # 去均值
    acf = torch.fft.irfft(torch.fft.rfft(x_centered, dim=1) * torch.conj(torch.fft.rfft(x_centered, dim=1)), dim=1)
    acf = acf[:, :T // 2, :]  # 只保留前半部分（对称）

    # 找到振幅峰值和对应的周期
    frequency_amplitudes = acf.mean(-1)  # 对 [C] 维度求平均
    frequency_amplitudes[:, 0] = 0  # 忽略滞后为 0 的值（自相关最大值）
    top_k = torch.topk(frequency_amplitudes, k, dim=1)  # 找到 k 个最大的自相关值

    periods = top_k.indices  # 对应的周期
    amplitudes = top_k.values  # 对应的自相关振幅

    return periods, amplitudes

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes * scale , kernel_size=1, padding=0)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, W = x.size()
        # N, W, C
        x_permuted = x.permute(0, 2, 1)
        # N,  W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W * self.scale , int(C / self.scale )))
        # N,C/(scale),W*scale
        x = x_permuted.permute(0, 2, 1)

        return x


class Downsampling(nn.Module):
    def __init__(self, in_ch, kernel_size):
        super(Downsampling, self).__init__()
        self.kernel_size = kernel_size
        self.wt = DWT1DForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Conv1d(in_ch * 2, in_ch, kernel_size=kernel_size , stride=kernel_size ,
                                      padding=kernel_size // 2)

    def forward(self, x):
        yL, yH = self.wt(x)
        x = torch.cat([yL, yH[0]], dim=1)  # 连接低频分量和高频分量
        x = self.conv_bn_relu(x)
        return x


class MCA(nn.Module):
    """
    MCA layer to extract local and global features
    """
    def __init__(self,top_k=3, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6],  device='cuda'):
        super(MCA, self).__init__()
        self.src_mask = None
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device
        self.top_k = top_k
        
        # causal convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                   kernel_size=(i+1)//2,padding=0,stride=1)
                                        for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i

        self.conv = nn.ModuleList([Downsampling(feature_size, i) for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([DUpsampling(feature_size,2*i) for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])

        self.fnn = FeedForwardNetwork(feature_size, feature_size*4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution 
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x+x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]   # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x


    def forward(self, src):
        B, T, N = src.size()
        period_list, period_weight = FFT_for_Period(src, self.top_k)
        # multi-scale
        multi = []  
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)  

        # adaptive aggregation
        multi = torch.stack(multi, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
             1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(multi * period_weight, -1)
        return self.fnn_norm(res + self.fnn(res))

class Seasonal_Prediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                conv_kernel=[2, 4], isometric_kernel=[18, 6],top_k=3, device='cuda'):
        super(Seasonal_Prediction, self).__init__()

        self.mic = nn.ModuleList([MCA(feature_size=embedding_size, n_heads=n_heads,
                                                   decomp_kernel=decomp_kernel,conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, top_k =top_k, device=device)
                                      for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)

