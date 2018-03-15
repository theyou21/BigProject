for j = 1:9
	printf('On dataset %i\n', j);
	eval(sprintf('[s,h] = sload(''A0%sT.gdf'');', int2str(j)))
	size_t=size(h.EVENT.TYP,1);
	type = zeros(1000,1);
	tc=1;
	image = zeros(1000,25);
	for i=1:size_t
		if h.EVENT.TYP(i)>=769 && h.EVENT.TYP(i)<=772;
			type(tc) = h.EVENT.TYP(i);
			if abs(h.EVENT.POS(i) - h.EVENT.POS(i-1) * 0.004 - 2) < 0.001;
				keyboard;
			end
			image(:,:,tc) = s(h.EVENT.POS(i)+1:h.EVENT.POS(i)+1000,:);
			tc=tc+1;
		   
		end
	end
	eval(sprintf('save A0%sT_slice.mat image type -mat7-binary', int2str(j)))

end
