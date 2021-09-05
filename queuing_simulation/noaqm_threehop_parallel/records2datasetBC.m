function train_records_bc = records2traindataBC(records)
    train_records_bc = [records(:,10)+records(:,7),records(:,14),records(:,15)];
end

