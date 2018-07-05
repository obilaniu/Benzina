function [p_hvs_m, p_hvs] = color_psnrhma(img1, img2)
  a=rgb2ycbcr(img1);
  a1=double(a(:,:,1));
  a2=double(a(:,:,2));
  a3=double(a(:,:,3));
  b=rgb2ycbcr(img2);
  b1=double(b(:,:,1));
  b2=double(b(:,:,2));
  b3=double(b(:,:,3));

  [p11,p12]=psnrhma(a1,b1);
  [p21,p22]=psnrhma(a2,b2);
  [p31,p32]=psnrhma(a3,b3);

  p11=unpsn(p11);p12=unpsn(p12);
  p21=unpsn(p21);p22=unpsn(p22);
  p31=unpsn(p31);p32=unpsn(p32);

  S1=(p11+(p21+p31)*0.5)/2;
  S2=(p12+(p22+p32)*0.5)/2;

  if S1 == 0 
    p_hvs_m = 100000; % img1 and img2 are visually undistingwished
  else
    p_hvs_m = 10*log10(255*255/S1);
  end
  if S2 == 0  
    p_hvs = 100000; % img1 and img2 are identical
  else
    p_hvs = 10*log10(255*255/S2);
  end

function d=unpsn(aa)
  d=255*255/(10^(aa/10));
