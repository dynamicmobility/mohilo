%% data loading
clear
clc
close all

load("KOS_walk_1_2_mean_cycle.mat")

WEIGHT = 75;

moment = mean_cycle.hip_flexion_moment;
samples_per_cycle = 500;
xq = linspace(0,1,samples_per_cycle);
samplingFreq = 455;
% level walking
lb = [0, 0,0,]; % lower scaling bounds
ub = [0.375, 0.3, 0.3]; % upper scaling bounds

paramDef = @(x, moment, power)...
    C1directionalTorqueDelayCombo(moment, samplingFreq, x(1), x(2), x(3));
c1_tp_outputs = GeneratePossibleCombos(lb, ub, paramDef, 5, moment, samplingFreq);

default_params = [0.1, 0.2, 0.2];
output_torque = C1directionalTorqueDelayCombo(moment, samplingFreq, default_params(1), default_params(2), default_params(3));
output_torque = mlutils.FiltFiltND(output_torque,samplingFreq,'FilterFreq',10,'Order',2);
n = numel(c1_tp_outputs);
cmap = cool(n); 
alpha = 0.75;

figure
hold on
max_curve = zeros(1,500);
min_curve = zeros(1,500);
for i = 1:numel(c1_tp_outputs)
    plot(xq, c1_tp_outputs{i}, 'Color', [cmap(i,:), alpha], 'LineWidth',1)
    tmp_curve = c1_tp_outputs{i};
    for j = 1:500
        if tmp_curve(j) > max_curve(j)
            max_curve(j) = tmp_curve(j);
        end
        if tmp_curve(j) < min_curve(j)
            min_curve(j) = tmp_curve(j);
        end

    end
end

hMoment = plot(xq, moment, 'LineWidth',2,'Color',[core.StandardColors("TCN_Blue"), 1]);
% hPower = plot(xq, power, 'LineWidth',2,'Color',[core.StandardColors("Maroon"), 1]);

hOpt = plot(xq, output_torque,'LineWidth',3,'Color',core.StandardColors("Black"));
hFill = fill([xq fliplr(xq)], [min_curve fliplr(max_curve)], [core.StandardColors("Violet")],'EdgeColor','none','FaceAlpha',0.2);
xlabel('Cycle (%)')
legend([hMoment, hOpt, hFill], ...
       {'Bio Joint Moment (Nm/kg)', 'Default Parameters','Combination Space'}, ...
       'Location', 'best')
title('Paramerization Space for 1.25 m/s walk')

%% control def
function outputTorque = C1directionalTorqueDelayCombo(moment, sampleFreq, delay, x_moment_pos, x_moment_neg)
    % should take delay in seconds and sampling freq in hz
    moment_direction = sign(moment);
    pos_idx = moment_direction>=0;
    neg_idx = moment_direction<0;
    outputTorque = NaN(length(moment),1);

    outputTorque(pos_idx) = x_moment_pos.*moment(pos_idx);
    outputTorque(neg_idx) = x_moment_neg.*moment(neg_idx);

    delay = floor(delay*sampleFreq);
    outputTorque = circshift(outputTorque, delay, 1);
end

function output_curves = GeneratePossibleCombos(lb, ub, paramDef, steps_per_param, varargin)
% GeneratePossibleCombos
%   lb, ub           : 1xN vectors
%   paramDef         : @(params, ...) -> output
%   steps_per_param  : scalar or 1xN vector
%   varargin         : extra fixed inputs forwarded to paramDef (in the
%   proper order)

    nParams = length(lb);

    if isscalar(steps_per_param)
        steps_per_param = repmat(steps_per_param, 1, nParams);
    end

    param_vec = cell(1, nParams);

    for i = 1:nParams
        param_vec{i} = linspace(lb(i), ub(i), steps_per_param(i));
    end

    [paramGrid{1:nParams}] = ndgrid(param_vec{:});

    nComb = numel(paramGrid{1});
    paramMat = zeros(nComb, nParams);
    for i = 1:nParams
        paramMat(:,i) = paramGrid{i}(:);
    end

    % ---- evaluate model ----
    output_curves = cell(nComb,1);
    if nComb < 100
        for k = 1:nComb
            output_curves{k} = paramDef(paramMat(k,:), varargin{:});
        end
    else
        parfor k = 1:nComb
            output_curves{k} = paramDef(paramMat(k,:), varargin{:});
        end
    end


end