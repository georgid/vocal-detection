function [NSDR,SIR, SAR ] = eval_svs_main(estim_iKalaURI, filename)

ref_iKalaURI = '/home/georgid/Documents/iKala/Wavfile/'


 
%%%%%%%%%%%%%%%%%%% reference
ref_fileURI = strcat(ref_iKalaURI, filename, '.wav');

[audio, fs] = audioread( ref_fileURI );
trueVoice = audio(:,2);
trueKaraoke = audio(:,1);

trueMixed = trueVoice + trueKaraoke;

%%%%%%%%%%%%%%%%%%%% estimated
estim_fileURI = strcat(estim_iKalaURI, filename, '_voice.wav');
[estimatedVoice_noPadding, fs] = audioread( estim_fileURI );
estimatedVoice = zeros(size(trueMixed));
diff_length = length(trueMixed) - length(estimatedVoice_noPadding);
estimatedVoice(diff_length+1:end) =  estimatedVoice_noPadding;

estimatedKaraoke_URI = strcat(estim_iKalaURI, filename, '_instr.wav');
[estimatedKaraoke, fs] = audioread( estimatedKaraoke_URI );

a = norm(estimatedVoice + estimatedKaraoke);
b =  norm(trueVoice + trueKaraoke);
[SDR, SIR, SAR] = bss_eval_sources([estimatedVoice estimatedKaraoke]' / a, [trueVoice trueKaraoke]' / b);
[NSDR, NSIR, NSAR] = bss_eval_sources([trueMixed trueMixed]' / norm(trueMixed + trueMixed), [trueVoice trueKaraoke]' / norm(trueVoice + trueKaraoke));

NSDR = SDR - NSDR;
NSIR = SIR - NSIR;
NSAR = SAR - NSAR;

end