
[C,ia,ic] = unique( records(:,[2 3 4]), 'rows' );
a_counts = accumarray(ic,1);
value_counts = [a_counts,a_counts/length(records), C];
value_counts = sortrows(value_counts,'descend');

fhm = value_counts(1:500,3:5);