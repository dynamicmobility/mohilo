% Parvo data processing
filePath = "D:\GaTech Dropbox\ME-DboxMgmt-Young-HIKETeam\MTJWO\MTBO_icra\MB04\MB04_2__0_20231223_1805.CSV";
% save_path = 
% filePath = "D:\GaTech Dropbox\ME-DboxMgmt-Young-AnkleExoTeam\Data\Combined State JW Opt\pilot_3_11\PILOT_ANKLE_00_0_20231212_1451.CSV";
mass = 89;
% mass = 70;
trialDuration = 300;
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

met = (0.278*trialData.vo2 + 0.075*trialData.vco2) / mass;
trialData = [trialData, array2table(met, 'VariableNames', {'met'})];


t = struct();

% Lift Weight
t.STAND = duration(0,6,00);
t.walk_val_1 = duration(0,13,21);
t.walk_val_2 = duration(0,18,47);
t.walk_val_3 = duration(0,24,23);
t.walk_val_4 = duration(0,29,54);
t.walk_val_5 = duration(0,35,21);
t.walk_val_6 = duration(0,40,50);

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

for ii=1:length(conditions)
    c = conditions{ii};
    startTime = seconds(t.(c));
    endTime = startTime + trialDuration;
    ssStartTime = endTime - steady_state_window;
    
    [~, endIdx] = min(abs(trialData.time - endTime));
    [~, startIdx] = min(abs(trialData.time - startTime));
    [~, ssStartIdx] = min(abs(trialData.time - ssStartTime));

%     metStruct.(c).avg = mean(trialData.met(ssStartIdx:endIdx));
%     metStruct.(c).ssdata = trialData.met(ssStartIdx:endIdx);
%     metStruct.(c).data = trialData.met(startIdx:endIdx);
%     metStruct.(c).time = trialData.time(startIdx:endIdx);

    cData = trialData(startIdx:endIdx, :);
    cDataSS = trialData(ssStartIdx:endIdx, :);

    metStruct.(c) = struct();
    metStruct.(c).avg = mean(cDataSS.met);
    metStruct.(c).std = std(cDataSS.met);
    metStruct.(c).ssdata = cDataSS.met;
    metStruct.(c).sstime = cDataSS.time;
    metStruct.(c).data = cData.met;
    metStruct.(c).time = cData.time;
    metStruct.(c).ssduration = diff(metStruct.(c).sstime);
    metStruct.(c).avgssduration = mean(metStruct.(c).ssduration);
    metStruct.(c).avgssrer = mean(cDataSS.rer);
    
    metStruct.(c).avgweighted = sum(metStruct.(c).ssduration .* metStruct.(c).ssdata(2:end)) / sum(metStruct.(c).ssduration);
    
%     h = plot(metStruct.(c).time, metStruct.(c).data, '-o');
% %     h = plot(metStruct.(c).sstime, metStruct.(c).ssdata, '-o');
%     color = get(h, 'Color');
% 
%     tsStartTime = find(~isnan(trialData.time));
%     tsEndTime = trialData.time(tsStartTime(end));
%     tsStartTime = trialData.time(tsStartTime(1));
%     plot([tsStartTime, tsEndTime], ones(1, 2) * metStruct.(c).avg, 'Color', color);
%     plot([tsStartTime, tsEndTime], ones(1, 2) * (metStruct.(c).avg - metStruct.(c).std), '--', 'Color', color);
%     plot([tsStartTime, tsEndTime], ones(1, 2) * (metStruct.(c).avg + metStruct.(c).std), '--', 'Color', color);
end

resultTable = table();
resultTable.trial = string(fieldnames(metStruct));
for i = 1:length(resultTable.trial)
resultTable.avg(i) = metStruct.(resultTable.trial{i}).avg - restingRate;
resultTable.std(i) = metStruct.(resultTable.trial{i}).std;
end

figure
hold on
bar(resultTable.trial, resultTable.avg,'interpreter','none')
errorbar(1:numel(resultTable.trial), resultTable.avg, resultTable.std,'LineStyle', 'none', 'LineWidth', 1.5,'Color','k')
title('MB04 Validation')


save('parvo_results.mat',"resultTable")

% incline_pchange = 100.*(avgResultTable.avg())

% plotTable = table();
% for ii=1:length(conditions)
%     c = conditions{ii};
%     plotTable = [plotTable, array2table(metStruct.(c).avgssrer, 'VariableNames', {c})];
% end
% 
% figure
% plot.barPlot(plotTable{:,:}, []);
% xticks(1:width(plotTable))
% xticklabels(plotTable.Properties.VariableNames)
% xtickangle(45)
% title('Avg RER')
% 
% plotTable = table();
% for ii=1:length(conditions)
%     c = conditions{ii};
%     plotTable = [plotTable, array2table(metStruct.(c).avgssduration, 'VariableNames', {c})];
% end
% 
% figure
% plot.barPlot(plotTable{:,:}, []);
% xticks(1:width(plotTable))
% xticklabels(plotTable.Properties.VariableNames)
% xtickangle(45)
% title('Avg Breath Duration')
% 
% plotTable = table();
% for ii=1:length(conditions)
%     c = conditions{ii};
%     plotTable = [plotTable, array2table(metStruct.(c).avg, 'VariableNames', {c})];
% end
% 
% figure
% plot.barPlot(plotTable{:,:}, []);
% xticks(1:width(plotTable))
% xticklabels(plotTable.Properties.VariableNames)
% xtickangle(45)
% 
% uniqueConditions = split(conditions, '_');
% uniqueConditions = unique(uniqueConditions(:, 1), 'stable');
% 
% plotTable = array2table(ones(length(conditions), length(uniqueConditions)) * NaN, 'VariableNames', uniqueConditions);
% for ii=1:length(uniqueConditions)
%     uc = uniqueConditions{ii};
% 
%     for jj=1:length(conditions)
%         c = conditions{jj};
%         if contains(c, uc)
%             plotTable.(uc)(jj) = metStruct.(c).avg;
%         end
%     end
% end
% 
% figure
% plot.barPlot(mean(plotTable{:,:}, 1, 'omitnan'), []);
% for ii=1:height(plotTable); plot(plotTable{ii, :}, 'og'); end
% xticks(1:length(uniqueConditions))
% xticklabels(uniqueConditions)
% xtickangle(45)
% title('Gross Metabolic Cost (W/kg)')
% 
% meanStand = mean(plotTable.STAND, 'omitnan');
% netTable = plotTable;
% netTable{:,:} = netTable{:,:} - meanStand;
% 
% figure
% plot.barPlot(mean(netTable{:,:}, 1, 'omitnan'), []);
% for ii=1:height(netTable); plot(netTable{ii, :}, 'og'); end
% xticks(1:length(uniqueConditions))
% xticklabels(uniqueConditions)
% xtickangle(45)
% title('Net Metabolic Cost (W/kg)')
% 
% meanNetDoff = mean(netTable.DOFF, 'omitnan');
% netDoffTable = netTable;
% netDoffTable{:,:} = (netDoffTable{:,:} - meanNetDoff) / meanNetDoff * 100;
% netDoffTable.STAND = [];
% 
% figure
% plot.barPlot(mean(netDoffTable{:,:}, 1, 'omitnan'), []);
% for ii=1:height(netDoffTable); plot(netDoffTable{ii, :}, 'og'); end
% xticks(1:length(uniqueConditions))
% xticklabels(uniqueConditions)
% xtickangle(45)
% title('Net Metabolic Cost W.R.T. DOFF (%)')
% 
% if any(strcmp(uniqueConditions, 'POFF'))
%     meanNetPoff = mean(netTable.POFF, 'omitnan');
%     netPoffTable = netTable;
%     netPoffTable{:,:} = (netPoffTable{:,:} - meanNetPoff) / meanNetPoff * 100;
%     netPoffTable.STAND = [];
% 
%     figure
%     plot.barPlot(mean(netPoffTable{:,:}, 1, 'omitnan'), []);
%     for ii=1:height(netPoffTable); plot(netPoffTable{ii, :}, 'og'); end
%     xticks(1:length(uniqueConditions))
%     xticklabels(uniqueConditions)
%     xtickangle(45)
%     title('Net Metabolic Cost W.R.T. POFF (%)')
% end
% 
% % meanNetPoff = mean(netTable.POFF, 'omitnan');
% % netPoffTable = netTable;
% % netPoffTable{:,:} = (netPoffTable{:,:} - meanNetPoff) / meanNetPoff * 100;
% % netPoffTable.STAND = [];
% % 
% % figure
% % plot.barPlot(mean(netPoffTable{:,:}, 1, 'omitnan'), []);
% % for ii=1:height(netPoffTable); plot(netPoffTable{ii, :}, 'og'); end
% % xticks(1:length(uniqueConditions))
% % xticklabels(uniqueConditions)
% % xtickangle(45)
% % title('Net Metabolic Cost W.R.T. POFF (%)')
% 
% %% Plot Steady-State Metabolic Timeseries
% h = 4;
% w = 4;
% figure
% subplot(h, w, 1)
% 
% conditons = fieldnames(t);
% % conditions(contains(conditions, 'STAND')) = [];
% 
% for ii=1:length(conditions)
%     c = conditions{ii};
%     subplot(h, w, ii)
%     plot(metStruct.(c).ssdata, '-o')
% %     plot(metStruct.(c).time - metStruct.(c).sstime(1), metStruct.(c).data, '-o')
% 
% %     ylim([0 5.5])
%     title(strrep(c, '_', '\_'))
% end
% 
% % %% Plot Avg Results with Weighting Based on Breath Duration
% % plotTable = array2table(ones(length(conditions), length(uniqueConditions)) * NaN, 'VariableNames', uniqueConditions);
% % for ii=1:length(uniqueConditions)
% %     uc = uniqueConditions{ii};
% %     
% %     for jj=1:length(conditions)
% %         c = conditions{jj};
% %         if contains(c, uc)
% %             plotTable.(uc)(jj) = metStruct.(c).avgweighted;
% %         end
% %     end
% % end
% % 
% % figure
% % plot.barPlot(mean(plotTable{:,:}, 1, 'omitnan'), []);
% % for ii=1:height(plotTable); plot(plotTable{ii, :}, 'og'); end
% % xticks(1:length(uniqueConditions))
% % xticklabels(uniqueConditions)
% % xtickangle(45)
% % title('Gross Metabolic Cost (W/kg)')
% % 
% % meanStand = mean(plotTable.STAND, 'omitnan');
% % netTable = plotTable;
% % netTable{:,:} = netTable{:,:} - meanStand;
% % 
% % figure
% % plot.barPlot(mean(netTable{:,:}, 1, 'omitnan'), []);
% % for ii=1:height(netTable); plot(netTable{ii, :}, 'og'); end
% % xticks(1:length(uniqueConditions))
% % xticklabels(uniqueConditions)
% % xtickangle(45)
% % title('Net Metabolic Cost (W/kg)')
% % 
% % meanNetDoff = mean(netTable.DOFF, 'omitnan');
% % netDoffTable = netTable;
% % netDoffTable{:,:} = (netDoffTable{:,:} - meanNetDoff) / meanNetDoff * 100;
% % netDoffTable.STAND = [];
% % 
% % figure
% % plot.barPlot(mean(netDoffTable{:,:}, 1, 'omitnan'), []);
% % for ii=1:height(netDoffTable); plot(netDoffTable{ii, :}, 'og'); end
% % xticks(1:length(uniqueConditions))
% % xticklabels(uniqueConditions)
% % xtickangle(45)
% % title('Net Metabolic Cost W.R.T. DOFF (%)')
% % 
% % if any(strcmp(uniqueConditions, 'POFF'))
% %     meanNetPoff = mean(netTable.POFF, 'omitnan');
% %     netPoffTable = netTable;
% %     netPoffTable{:,:} = (netPoffTable{:,:} - meanNetPoff) / meanNetPoff * 100;
% %     netPoffTable.STAND = [];
% % 
% %     figure
% %     plot.barPlot(mean(netPoffTable{:,:}, 1, 'omitnan'), []);
% %     for ii=1:height(netPoffTable); plot(netPoffTable{ii, :}, 'og'); end
% %     xticks(1:length(uniqueConditions))
% %     xticklabels(uniqueConditions)
% %     xtickangle(45)
% %     title('Net Metabolic Cost W.R.T. POFF (%)')
% % end
% 
% %%
% figure
% plot.barPlot(mean(netTable{:,:}, 1, 'omitnan'), []);
% for ii=1:height(netTable); plot(netTable{ii, :}, 'og'); end
% xticks(1:length(uniqueConditions))
% xticklabels(uniqueConditions)
% xtickangle(45)
% title('Net Metabolic Cost (W/kg)')
% 
% plotVars = {'DOFF', 'BT', 'POFF'};
% pTable = netTable(:, plotVars);
% 
% f = figure;
% fig=gcf;
% fig.PaperUnits = 'inches';
% fig.PaperPosition = [0 0 5.5 3.5];
% 
% plot.barPlot(mean(pTable{:,:}, 1, 'omitnan'), []);
% for ii=1:height(pTable); plot(pTable{ii, :}, 'og'); end
% xticks(1:length(plotVars))
% % xticklabels(plotVars)
% xticklabels({'No Exo', 'Exo', 'Powered Off'})
% xtickangle(45)
% title('Level Ground Walking at 1.25 m/s')
% ylabel('Avg. Net Metabolic Cost (W/kg)')
% box off
% 
% saveas(f, [saveDir '\LG_ssMetNet.png']);