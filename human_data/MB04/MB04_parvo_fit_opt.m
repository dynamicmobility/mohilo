% Parvo data processing
filePath = "D:\GaTech Dropbox\ME-DboxMgmt-Young-HIKETeam\MTJWO\MTBO_icra\MB04\MB04__0_20231223_0713.CSV";
% save_path = 
% filePath = "D:\GaTech Dropbox\ME-DboxMgmt-Young-AnkleExoTeam\Data\Combined State JW Opt\pilot_3_11\PILOT_ANKLE_00_0_20231212_1451.CSV";
mass = 89;
% mass = 70;
trialDuration = 180;
stand_duration = 120;
% opt_duration = 156;
% val_duration = 300;
steady_state_window = 60;

h = {'time', 'vo2', 'vo2_kg', 'mets', 'vco2', 've', 'rer', 'rr', 'vt', 'feo2', 'feco2', 'ackcal'};
opts = detectImportOptions(filePath, 'NumHeaderLines', 29);
trialData = readtable(filePath, opts);
trialData.Properties.VariableNames = h;

trialData.vo2 = trialData.vo2 * 1000; % mL/min
trialData.vco2 = trialData.vco2 * 1000; % mL/min
trialData.time = trialData.time * 60; % seconds

met = (0.278*trialData.vo2 + 0.075*trialData.vco2) / mass; % brockway eq
trialData = [trialData, array2table(met, 'VariableNames', {'met'})];


t = struct();

% Input times from experiment
t.STAND = duration(0,1,01);
t.walk_opt_1 = duration(0,7,59);
t.walk_opt_2 = duration(0,11,49);
t.walk_opt_3 = duration(0,15,07);
t.walk_opt_4 = duration(0,18,27);
t.walk_opt_5 = duration(0,49,38);
t.walk_opt_6 = duration(0,53,02);
t.walk_opt_7 = duration(0,56,46);
t.walk_opt_8 = duration(0,59,47);
t.walk_opt_9 = duration(0,63,05);
t.walk_opt_10 = duration(0,66,34);
t.walk_opt_11 = duration(0,69,46);
t.walk_opt_12 = duration(0,73,21);
t.walk_opt_13 = duration(0,91,42);
t.walk_opt_14 = duration(0,95,04);
t.walk_opt_15 = duration(0,104,41);
t.walk_opt_16 = duration(0,108,02);
t.walk_opt_17 = duration(0,117,18);
t.walk_opt_18 = duration(0,120,36);
t.walk_opt_19 = duration(0,123,53);
t.walk_opt_20 = duration(0,127,23);
t.walk_opt_21 = duration(0,130,42);
t.walk_opt_22 = duration(0,134,00);
t.walk_opt_23 = duration(0,137,20);
t.walk_opt_24 = duration(0,140,33);

conditions = fieldnames(t);

metStruct = struct();

figure
plot(trialData.rer, '-o')
title('RER')

conditions = conditions(contains(conditions,'walk'));

resting_start = seconds(t.STAND);
resting_end = resting_start + 120;
[~, endIdx] = min(abs(trialData.time - resting_end));
[~, startIdx] = min(abs(trialData.time - resting_start));
restingData = trialData(startIdx:endIdx, :);
restingRate = mean(restingData.met);

ft = fittype('a + b*exp(-x/c)','independent','x','coefficients',{'a','b','c'});
for ii=1:length(conditions)
    c = conditions{ii};
    startTime = seconds(t.(c));
    endTime = startTime + trialDuration;
    ssStartTime = endTime - steady_state_window;
    
    [~, endIdx] = min(abs(trialData.time - endTime));
    [~, startIdx] = min(abs(trialData.time - startTime));
    [~, ssStartIdx] = min(abs(trialData.time - ssStartTime));

    cData = trialData(startIdx:endIdx, :);
    cDataSS = trialData(ssStartIdx:endIdx, :);

    opts = fitoptions( ...
        Method='NonlinearLeastSquares', ...
        StartPoint=[cData.met(end), cData.met(1)-cData.met(end), 60], ...
        Lower=[-Inf, -Inf, 0], ...
        Upper=[Inf, Inf, 52]);  % rise time is bounded to something realistic )
    fitResult = fit(cData.time - cData.time(1), cData.met, ft, opts);
    avgEstMetPower = (1/(360 - 240)) * integral(@(t) fitResult(t), 240, 360, ArrayValued=true);

    metStruct.(c) = struct();
    metStruct.(c).avg = mean(cDataSS.met);
    metStruct.(c).estimate = avgEstMetPower;
    metStruct.(c).fit = fitResult;
    metStruct.(c).std = std(cDataSS.met);
    metStruct.(c).ssdata = cDataSS.met;
    metStruct.(c).sstime = cDataSS.time;
    metStruct.(c).data = cData.met;
    metStruct.(c).time = cData.time;
    metStruct.(c).ssduration = diff(metStruct.(c).sstime);
    metStruct.(c).avgssduration = mean(metStruct.(c).ssduration);
    metStruct.(c).avgssrer = mean(cDataSS.rer);
    
    metStruct.(c).avgweighted = sum(metStruct.(c).ssduration .* metStruct.(c).ssdata(2:end)) / sum(metStruct.(c).ssduration);
    
    % h = plot(metStruct.(c).time, metStruct.(c).data, '-o');
    % % h = plot(metStruct.(c).sstime, metStruct.(c).ssdata, '-o');
    % color = get(h, 'Color');
    % 
    % tsStartTime = find(~isnan(trialData.time));
    % tsEndTime = trialData.time(tsStartTime(end));
    % tsStartTime = trialData.time(tsStartTime(1));
    % plot([tsStartTime, tsEndTime], ones(1, 2) * metStruct.(c).avg, 'Color', color);
    % plot([tsStartTime, tsEndTime], ones(1, 2) * (metStruct.(c).avg - metStruct.(c).std), '--', 'Color', color);
    % plot([tsStartTime, tsEndTime], ones(1, 2) * (metStruct.(c).avg + metStruct.(c).std), '--', 'Color', color);
end

resultTable = table();
resultTable.trial = string(fieldnames(metStruct));
for i = 1:length(resultTable.trial)
resultTable.avg(i) = metStruct.(resultTable.trial{i}).avg - restingRate;
resultTable.estimate(i) = metStruct.(resultTable.trial{i}).estimate - restingRate;
resultTable.std(i) = metStruct.(resultTable.trial{i}).std;
end


save('parvo_results_opt.mat',"resultTable")

%% cost visualization
figure
hold on
bar(resultTable.trial, resultTable.estimate,'interpreter','none')
% errorbar(1:numel(resultTable.trial), resultTable.estimate, resultTable.std,'LineStyle', 'none', 'LineWidth', 1.5,'Color','k')
title('MB04 Optimization')


%% fit visuzalization
trial = 'walk_opt_8';

figure
hold on
scatter(metStruct.(trial).time - metStruct.(trial).time(1),  metStruct.(trial).data,'b')
plot(metStruct.(trial).fit)
yline(metStruct.(trial).estimate,'Color','k')
xlabel('Time (s)')
ylabel('Metabolic Rate (W/kg)')
legend('Raw Data','1st order fit','Estimated Steady-State Rate')