records = table2array(recordsTable);

datasetBU = records2datasetBU(records);
[C,ia,ic] = unique(datasetBU(:,2:4),'rows');
value_counts = [C,accumarray(ic,1)];
most_common_BU = sortrows(value_counts,4,'descend');
most_common_BU = [most_common_BU,most_common_BU(:,4)/size(datasetBU,1)];
most_common_BU = [most_common_BU,cumsum(most_common_BU(:,5))];

datasetBC = records2datasetBC(records);
[C,ia,ic] = unique(datasetBC(:,2:3),'rows');
value_counts = [C,accumarray(ic,1)];
most_common_BC = sortrows(value_counts,3,'descend');
most_common_BC = [most_common_BC,most_common_BC(:,3)/size(datasetBC,1)];
most_common_BC = [most_common_BC,cumsum(most_common_BC(:,4))];

datasetBD = records2datasetBD(records);
[C,ia,ic] = unique(datasetBD(:,2),'rows');
value_counts = [C,accumarray(ic,1)];
most_common_BD = sortrows(value_counts,2,'descend');
most_common_BD = [most_common_BD,most_common_BD(:,2)/size(datasetBD,1)];
most_common_BD = [most_common_BD,cumsum(most_common_BD(:,3))];

clear 'C' 'ia' 'ic' 'value_counts';