mv GRU_ADAM-18509691.out PB4_2/GRU_ADAM_log.txt
mv 'GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0' PB4_2/
mv GRU_SGD-18509693.out PB4_2/GRU_SGD.txt
mv 'GRU_SGD_model=GRU_optimizer=SGD_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0' PB4_2/

mv TRANSFORMER_ADAM-18513610.out PB4_2/TRANSFORMER_ADAM_log.txt
mv 'TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=.9_save_best_0' PB4_2/
mv TRANSFORMER_SGD-18513614.out PB4_2/TRANSFORMER_SGD_log.txt
mv 'TRANSFORMER_SGD_model=TRANSFORMER_optimizer=SGD_initial_lr=20_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=.9_save_best_0' PB4_2/

mv RNN_ADAM-18506753.out RNN_ADAM_log.txt
mv 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0' PB4_1/
