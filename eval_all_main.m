function  eval_all_main()
iKala_estim_URI = '/home/georgid/Documents/iKala/Wavfile_resynth/';
filenames = {'45416_verse', '45412_chorus', '54247_verse', '10161_chorus', '10161_verse', '10170_chorus',  '31113_chorus',   '45412_verse'}
a = strcat(iKala_estim_URI,  '*.wav');
% filenames = dir(a);

NSDR_total = 0
SIR_total = 0
SAR_total = 0

for i=1:length(filenames)
    
 %   [NSDR, SIR, SAR ] = eval_svs_main(iKala_estim_URI, filenames(i).name)
    [NSDR, SIR, SAR ] = eval_svs_main(iKala_estim_URI, filenames{i});
    NSDR_total = NSDR_total +  NSDR;
    SIR_total = SIR_total + SIR;
    SAR_total = SAR_total + SAR;
end

nsdr = NSDR_total  / length(filenames)
sir = SIR_total / length(filenames)
sar =  SAR_total /  length(filenames)

disp('nsdr:'); disp(nsdr)
disp('sir:'); disp(sir)
disp('sar:'); disp(sar)


end