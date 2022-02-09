import os



feature_selection = 'erp_lowered'
filter = "False"

total_rank_list = {}
re_list = {}
for f_i in range(21):
    total_rank_list[str(f_i)], re_list[str(f_i)] = get_file(f_i, parse_result.feature_selection, parse_result.filter)
fw = open('tmp_data/' + parse_result.feature_selection + '_' + str(parse_result.filter),'w')
fw.write(json.dumps(total_rank_list))
fw.write('\n')
fw.write(json.dumps(re_list))
fw.close()
