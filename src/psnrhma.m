function [p_hvs_m, p_hvs] = psnrhma(img1, img2, wstep)

%========================================================================
%
% Calculation of PSNR-HMA and PSNR-HA image quality measures
%
% PSNR-HMA is Peak Signal to Noise Ratio taking into account 
% Contrast Sensitivity Function (CSF), between-coefficient   
% contrast masking of DCT basis functions as well as mean shift and
% contrast changing
% PSNR-HA is the same metric without taking into account constrast masking 
%
% Copyright(c) 2011 Nikolay Ponomarenko 
% All Rights Reserved
%
% Homepage: www.ponomarenko.info , E-mail: nikolay@ponomarenko.info
%
%----------------------------------------------------------------------
%
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the PSNR-HA
% or PSNR-HMA between two images. Please refer to the following paper:
%
% N. Ponomarenko, O. Ieremeiev, V. Lukin, K. Egiazarian, M. Carli, 
% Modified Image Visual Quality Metrics for Contrast Change and Mean Shift 
% Accounting, Proceedings of CADSM, February 2011, Ukraine, pp. 305 - 311. 
%
% Kindly report any suggestions or corrections to nikolay@ponomarenko.info
%
%----------------------------------------------------------------------
%
% Input : (1) img1: the first image being compared
%         (2) img2: the second image being compared
%         (3) wstep: step of 8x8 window to calculate DCT 
%             coefficients. Default value is 8.
%
% Output: (1) p_hvs_m: the PSNR-HMA value between 2 images.
%             If one of the images being compared is regarded as 
%             perfect quality, then PSNR-HMA can be considered as the
%             quality measure of the other image.
%             If compared images are visually undistingwished, 
%             then PSNR-HMA = 100000.
%         (2) p_hvs: the PSNR-HA value between 2 images.
%
% Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [p_hvs_m, p_hvs] = psnrhvsm(img1, img2);
%
% See the results:
%
%   p_hvs_m  % Gives the PSNR-HMA value
%   p_hvs    % Gives the PSNR-HA value
%
%========================================================================

if nargin < 2
  p_hvs_m = -Inf;
  p_hvs = -Inf;
  return;
end

if size(img1) ~= size(img2)
  p_hvs_m = -Inf;
  p_hvs = -Inf;
  return;
end

if nargin > 2 
  step = wstep;
else
  step = 8; % Default value is 8;
end

LenXY=size(img1);LenY=LenXY(1);LenX=LenXY(2);

CSFCof  = [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887;
           2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911;
           1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555;
           1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082;
           1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222;
           1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729;
           0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803;
           0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950];
% see an explanation in [2]

MaskCof = [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874;
           0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058;
           0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888;
           0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015;
           0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866;
           0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815;
           0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803;
           0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203];
% see an explanation in [1]


delt=(sum(double(img1(:)))-sum(double(img2(:))))/length(img1(:));
img2m=double(img2)+delt;

mean1=mean(img1(:));
mean2=mean(img2m(:));
tmp=(img1-mean1).*(img2m-mean2);
sq=var(img2m(:),1);
l=sum(tmp(:))/length(tmp(:))/sq;
if l<1
  KofContr=0.002;
else
  KofContr=0.25;
end;
img3m=mean2+(img2m-mean2)*l;
S3=(sum(sum((img2m-img1).^2))-sum(sum((img3m-img1).^2)))/length(img1(:));

S1 = 0; S2 = 0; Num = 0;
SS1=0;SS2=0;
X=1;Y=1;
while Y <= LenY-7
  while X <= LenX-7 
    A = img1(Y:Y+7,X:X+7);
    B = img2m(Y:Y+7,X:X+7);
    B2 = img3m(Y:Y+7,X:X+7);
    A_dct = dct2(A); B_dct = dct2(B); B_dct2 = dct2(B2);
    MaskA = maskeff(A,A_dct,X,Y);
    MaskB = maskeff(B,B_dct,X,Y);
    if MaskB > MaskA
      MaskA = MaskB;
    end
    X = X + step;
    for k = 1:8
      for l = 1:8
        u   = abs(A_dct(k,l) - B_dct (k,l));
        u2  = abs(A_dct(k,l) - B_dct2(k,l));
        S2  = S2  + (u *CSFCof(k,l)).^2;    % PSNR-HVS
        SS2 = SS2 + (u2*CSFCof(k,l)).^2;    % PSNR-HVS

        if (k~=1) || (l~=1)               % See equation 3 in [1]
          if u < MaskA/MaskCof(k,l) 
            u = 0;
          else
            u = u - MaskA/MaskCof(k,l);
          end;
          if u2 < MaskA/MaskCof(k,l) 
            u2 = 0;
          else
            u2 = u2 - MaskA/MaskCof(k,l);
          end;
        end
        S1  = S1  + (u *CSFCof(k,l)).^2;    % PSNR-HVS-M
        SS1 = SS1 + (u2*CSFCof(k,l)).^2;    % PSNR-HVS-M
        Num = Num + 1;
      end
    end
  end
  X = 1; Y = Y + step;
end

if Num ~=0
  S1  = S1 /Num;
  S2  = S2 /Num;
  SS1 = SS1/Num;
  SS2 = SS2/Num;
  delt=delt^2;

  if (S1>SS1) S1=SS1+(S1-SS1)*KofContr;
  end
  S1=S1 + 0.04*delt;
  if (S2>SS2) S2=SS2+(S2-SS2)*KofContr;
  end
  S2=S2 + 0.04*delt;
  
  if S1 == 0
    p_hvs_m = 100000; % img1 and img2 are visually undistingwished
  else
    p_hvs_m = 10*log10(255*255/S1);
  end
  if S2 == 0
    p_hvs   = 100000; % img1 and img2 are identical
  else
    p_hvs   = 10*log10(255*255/S2);
  end
end

function m=maskeff(z,zdct,X,Y)  
% Calculation of Enorm value (see [1])
m = 0;

MaskCof = [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874;
           0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058;
           0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888;
           0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015;
           0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866;
           0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815;
           0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803;
           0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203];
% see an explanation in [1]

for k = 1:8
  for l = 1:8
    if (k~=1) || (l~=1)
      m = m + (zdct(k,l).^2) * MaskCof(k,l);
    end
  end
end
pop=vari(z);
if pop ~= 0
  pop=(vari(z(1:4,1:4))+vari(z(1:4,5:8))+vari(z(5:8,5:8))+vari(z(5:8,1:4)))/pop;
end
#X = int32(fix(X/8));
#Y = int32(fix(Y/8));
#printf("(%2d,%2d) %.20f\n", Y, X, sqrt(m*pop)/32)
m = sqrt(m*pop)/32;   % sqrt(m*pop/16/64)

function d=vari(AA)
  d=var(AA(:))*length(AA(:));
