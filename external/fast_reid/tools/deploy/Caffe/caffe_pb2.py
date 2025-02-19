

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database


_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='caffe.proto',
  package='caffe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x0b\x63\x61\x66\x66\x65.proto\x12\x05\x63\x61\x66\x66\x65\"\x1c\n\tBlobShape\x12\x0f\n\x03\x64im\x18\x01 \x03(\x03\x42\x02\x10\x01\"\xcc\x01\n\tBlobProto\x12\x1f\n\x05shape\x18\x07 \x01(\x0b\x32\x10.caffe.BlobShape\x12\x10\n\x04\x64\x61ta\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\x10\n\x04\x64iff\x18\x06 \x03(\x02\x42\x02\x10\x01\x12\x17\n\x0b\x64ouble_data\x18\x08 \x03(\x01\x42\x02\x10\x01\x12\x17\n\x0b\x64ouble_diff\x18\t \x03(\x01\x42\x02\x10\x01\x12\x0e\n\x03num\x18\x01 \x01(\x05:\x01\x30\x12\x13\n\x08\x63hannels\x18\x02 \x01(\x05:\x01\x30\x12\x11\n\x06height\x18\x03 \x01(\x05:\x01\x30\x12\x10\n\x05width\x18\x04 \x01(\x05:\x01\x30\"2\n\x0f\x42lobProtoVector\x12\x1f\n\x05\x62lobs\x18\x01 \x03(\x0b\x32\x10.caffe.BlobProto\"\x91\x01\n\x05\x44\x61tum\x12\x10\n\x08\x63hannels\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\x12\r\n\x05label\x18\x05 \x01(\x05\x12\x12\n\nfloat_data\x18\x06 \x03(\x02\x12\x16\n\x07\x65ncoded\x18\x07 \x01(\x08:\x05\x66\x61lse\x12\x0e\n\x06labels\x18\x08 \x03(\x02\"A\n\x0cLabelMapItem\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05label\x18\x02 \x01(\x05\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\"-\n\x08LabelMap\x12!\n\x04item\x18\x01 \x03(\x0b\x32\x13.caffe.LabelMapItem\"o\n\x07Sampler\x12\x14\n\tmin_scale\x18\x01 \x01(\x02:\x01\x31\x12\x14\n\tmax_scale\x18\x02 \x01(\x02:\x01\x31\x12\x1b\n\x10min_aspect_ratio\x18\x03 \x01(\x02:\x01\x31\x12\x1b\n\x10max_aspect_ratio\x18\x04 \x01(\x02:\x01\x31\"\xc0\x01\n\x10SampleConstraint\x12\x1b\n\x13min_jaccard_overlap\x18\x01 \x01(\x02\x12\x1b\n\x13max_jaccard_overlap\x18\x02 \x01(\x02\x12\x1b\n\x13min_sample_coverage\x18\x03 \x01(\x02\x12\x1b\n\x13max_sample_coverage\x18\x04 \x01(\x02\x12\x1b\n\x13min_object_coverage\x18\x05 \x01(\x02\x12\x1b\n\x13max_object_coverage\x18\x06 \x01(\x02\"\xb2\x01\n\x0c\x42\x61tchSampler\x12 \n\x12use_original_image\x18\x01 \x01(\x08:\x04true\x12\x1f\n\x07sampler\x18\x02 \x01(\x0b\x32\x0e.caffe.Sampler\x12\x32\n\x11sample_constraint\x18\x03 \x01(\x0b\x32\x17.caffe.SampleConstraint\x12\x12\n\nmax_sample\x18\x04 \x01(\r\x12\x17\n\nmax_trials\x18\x05 \x01(\r:\x03\x31\x30\x30\"\x8a\x01\n\x0e\x45mitConstraint\x12\x39\n\temit_type\x18\x01 \x01(\x0e\x32\x1e.caffe.EmitConstraint.EmitType:\x06\x43\x45NTER\x12\x14\n\x0c\x65mit_overlap\x18\x02 \x01(\x02\"\'\n\x08\x45mitType\x12\n\n\x06\x43\x45NTER\x10\x00\x12\x0f\n\x0bMIN_OVERLAP\x10\x01\"\x87\x01\n\x0eNormalizedBBox\x12\x0c\n\x04xmin\x18\x01 \x01(\x02\x12\x0c\n\x04ymin\x18\x02 \x01(\x02\x12\x0c\n\x04xmax\x18\x03 \x01(\x02\x12\x0c\n\x04ymax\x18\x04 \x01(\x02\x12\r\n\x05label\x18\x05 \x01(\x05\x12\x11\n\tdifficult\x18\x06 \x01(\x08\x12\r\n\x05score\x18\x07 \x01(\x02\x12\x0c\n\x04size\x18\x08 \x01(\x02\"I\n\nAnnotation\x12\x16\n\x0binstance_id\x18\x01 \x01(\x05:\x01\x30\x12#\n\x04\x62\x62ox\x18\x02 \x01(\x0b\x32\x15.caffe.NormalizedBBox\"M\n\x0f\x41nnotationGroup\x12\x13\n\x0bgroup_label\x18\x01 \x01(\x05\x12%\n\nannotation\x18\x02 \x03(\x0b\x32\x11.caffe.Annotation\"\xaf\x01\n\x0e\x41nnotatedDatum\x12\x1b\n\x05\x64\x61tum\x18\x01 \x01(\x0b\x32\x0c.caffe.Datum\x12\x32\n\x04type\x18\x02 \x01(\x0e\x32$.caffe.AnnotatedDatum.AnnotationType\x12\x30\n\x10\x61nnotation_group\x18\x03 \x03(\x0b\x32\x16.caffe.AnnotationGroup\"\x1a\n\x0e\x41nnotationType\x12\x08\n\x04\x42\x42OX\x10\x00\"C\n\tMTCNNBBox\x12\x0c\n\x04xmin\x18\x01 \x01(\x02\x12\x0c\n\x04ymin\x18\x02 \x01(\x02\x12\x0c\n\x04xmax\x18\x03 \x01(\x02\x12\x0c\n\x04ymax\x18\x04 \x01(\x02\"U\n\nMTCNNDatum\x12\x1b\n\x05\x64\x61tum\x18\x01 \x01(\x0b\x32\x0c.caffe.Datum\x12\x1d\n\x03roi\x18\x02 \x01(\x0b\x32\x10.caffe.MTCNNBBox\x12\x0b\n\x03pts\x18\x03 \x03(\x02\"\x98\x02\n\x0f\x46illerParameter\x12\x16\n\x04type\x18\x01 \x01(\t:\x08\x63onstant\x12\x10\n\x05value\x18\x02 \x01(\x02:\x01\x30\x12\x0e\n\x03min\x18\x03 \x01(\x02:\x01\x30\x12\x0e\n\x03max\x18\x04 \x01(\x02:\x01\x31\x12\x0f\n\x04mean\x18\x05 \x01(\x02:\x01\x30\x12\x0e\n\x03std\x18\x06 \x01(\x02:\x01\x31\x12\x12\n\x06sparse\x18\x07 \x01(\x05:\x02-1\x12\x42\n\rvariance_norm\x18\x08 \x01(\x0e\x32#.caffe.FillerParameter.VarianceNorm:\x06\x46\x41N_IN\x12\x0c\n\x04\x66ile\x18\t \x01(\t\"4\n\x0cVarianceNorm\x12\n\n\x06\x46\x41N_IN\x10\x00\x12\x0b\n\x07\x46\x41N_OUT\x10\x01\x12\x0b\n\x07\x41VERAGE\x10\x02\"\x8e\x02\n\x0cNetParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05input\x18\x03 \x03(\t\x12%\n\x0binput_shape\x18\x08 \x03(\x0b\x32\x10.caffe.BlobShape\x12\x11\n\tinput_dim\x18\x04 \x03(\x05\x12\x1d\n\x0e\x66orce_backward\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x1e\n\x05state\x18\x06 \x01(\x0b\x32\x0f.caffe.NetState\x12\x19\n\ndebug_info\x18\x07 \x01(\x08:\x05\x66\x61lse\x12$\n\x05layer\x18\x64 \x03(\x0b\x32\x15.caffe.LayerParameter\x12\'\n\x06layers\x18\x02 \x03(\x0b\x32\x17.caffe.V1LayerParameter\"\xc0\n\n\x0fSolverParameter\x12\x0b\n\x03net\x18\x18 \x01(\t\x12&\n\tnet_param\x18\x19 \x01(\x0b\x32\x13.caffe.NetParameter\x12\x11\n\ttrain_net\x18\x01 \x01(\t\x12\x10\n\x08test_net\x18\x02 \x03(\t\x12,\n\x0ftrain_net_param\x18\x15 \x01(\x0b\x32\x13.caffe.NetParameter\x12+\n\x0etest_net_param\x18\x16 \x03(\x0b\x32\x13.caffe.NetParameter\x12$\n\x0btrain_state\x18\x1a \x01(\x0b\x32\x0f.caffe.NetState\x12#\n\ntest_state\x18\x1b \x03(\x0b\x32\x0f.caffe.NetState\x12\x11\n\ttest_iter\x18\x03 \x03(\x05\x12\x18\n\rtest_interval\x18\x04 \x01(\x05:\x01\x30\x12 \n\x11test_compute_loss\x18\x13 \x01(\x08:\x05\x66\x61lse\x12!\n\x13test_initialization\x18  \x01(\x08:\x04true\x12\x0f\n\x07\x62\x61se_lr\x18\x05 \x01(\x02\x12\x0f\n\x07\x64isplay\x18\x06 \x01(\x05\x12\x17\n\x0c\x61verage_loss\x18! \x01(\x05:\x01\x31\x12\x10\n\x08max_iter\x18\x07 \x01(\x05\x12\x14\n\titer_size\x18$ \x01(\x05:\x01\x31\x12\x11\n\tlr_policy\x18\x08 \x01(\t\x12\r\n\x05gamma\x18\t \x01(\x02\x12\r\n\x05power\x18\n \x01(\x02\x12\x10\n\x08momentum\x18\x0b \x01(\x02\x12\x14\n\x0cweight_decay\x18\x0c \x01(\x02\x12\x1f\n\x13regularization_type\x18\x1d \x01(\t:\x02L2\x12\x10\n\x08stepsize\x18\r \x01(\x05\x12\x11\n\tstepvalue\x18\" \x03(\x05\x12\x0f\n\x07stagelr\x18\x32 \x03(\x02\x12\x11\n\tstageiter\x18\x33 \x03(\x05\x12\x1a\n\x0e\x63lip_gradients\x18# \x01(\x02:\x02-1\x12\x13\n\x08snapshot\x18\x0e \x01(\x05:\x01\x30\x12\x17\n\x0fsnapshot_prefix\x18\x0f \x01(\t\x12\x1c\n\rsnapshot_diff\x18\x10 \x01(\x08:\x05\x66\x61lse\x12K\n\x0fsnapshot_format\x18% \x01(\x0e\x32%.caffe.SolverParameter.SnapshotFormat:\x0b\x42INARYPROTO\x12;\n\x0bsolver_mode\x18\x11 \x01(\x0e\x32!.caffe.SolverParameter.SolverMode:\x03GPU\x12\x14\n\tdevice_id\x18\x12 \x01(\x05:\x01\x30\x12\x17\n\x0brandom_seed\x18\x14 \x01(\x03:\x02-1\x12\x11\n\x04type\x18( \x01(\t:\x03SGD\x12\x14\n\x05\x64\x65lta\x18\x1f \x01(\x02:\x05\x31\x65-08\x12\x18\n\tmomentum2\x18\' \x01(\x02:\x05\x30.999\x12\x11\n\trms_decay\x18& \x01(\x02\x12\x19\n\ndebug_info\x18\x17 \x01(\x08:\x05\x66\x61lse\x12\"\n\x14snapshot_after_train\x18\x1c \x01(\x08:\x04true\x12;\n\x0bsolver_type\x18\x1e \x01(\x0e\x32!.caffe.SolverParameter.SolverType:\x03SGD\"+\n\x0eSnapshotFormat\x12\x08\n\x04HDF5\x10\x00\x12\x0f\n\x0b\x42INARYPROTO\x10\x01\"\x1e\n\nSolverMode\x12\x07\n\x03\x43PU\x10\x00\x12\x07\n\x03GPU\x10\x01\"U\n\nSolverType\x12\x07\n\x03SGD\x10\x00\x12\x0c\n\x08NESTEROV\x10\x01\x12\x0b\n\x07\x41\x44\x41GRAD\x10\x02\x12\x0b\n\x07RMSPROP\x10\x03\x12\x0c\n\x08\x41\x44\x41\x44\x45LTA\x10\x04\x12\x08\n\x04\x41\x44\x41M\x10\x05\"l\n\x0bSolverState\x12\x0c\n\x04iter\x18\x01 \x01(\x05\x12\x13\n\x0blearned_net\x18\x02 \x01(\t\x12!\n\x07history\x18\x03 \x03(\x0b\x32\x10.caffe.BlobProto\x12\x17\n\x0c\x63urrent_step\x18\x04 \x01(\x05:\x01\x30\"N\n\x08NetState\x12!\n\x05phase\x18\x01 \x01(\x0e\x32\x0c.caffe.Phase:\x04TEST\x12\x10\n\x05level\x18\x02 \x01(\x05:\x01\x30\x12\r\n\x05stage\x18\x03 \x03(\t\"s\n\x0cNetStateRule\x12\x1b\n\x05phase\x18\x01 \x01(\x0e\x32\x0c.caffe.Phase\x12\x11\n\tmin_level\x18\x02 \x01(\x05\x12\x11\n\tmax_level\x18\x03 \x01(\x05\x12\r\n\x05stage\x18\x04 \x03(\t\x12\x11\n\tnot_stage\x18\x05 \x03(\t\"\x90\x02\n\x1bSpatialTransformerParameter\x12\x1e\n\x0etransform_type\x18\x01 \x01(\t:\x06\x61\x66\x66ine\x12\x1e\n\x0csampler_type\x18\x02 \x01(\t:\x08\x62ilinear\x12\x10\n\x08output_H\x18\x03 \x01(\x05\x12\x10\n\x08output_W\x18\x04 \x01(\x05\x12\x1b\n\rto_compute_dU\x18\x05 \x01(\x08:\x04true\x12\x11\n\ttheta_1_1\x18\x06 \x01(\x01\x12\x11\n\ttheta_1_2\x18\x07 \x01(\x01\x12\x11\n\ttheta_1_3\x18\x08 \x01(\x01\x12\x11\n\ttheta_2_1\x18\t \x01(\x01\x12\x11\n\ttheta_2_2\x18\n \x01(\x01\x12\x11\n\ttheta_2_3\x18\x0b \x01(\x01\"5\n\x0fSTLossParameter\x12\x10\n\x08output_H\x18\x01 \x02(\x05\x12\x10\n\x08output_W\x18\x02 \x02(\x05\"\xa3\x01\n\tParamSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x31\n\nshare_mode\x18\x02 \x01(\x0e\x32\x1d.caffe.ParamSpec.DimCheckMode\x12\x12\n\x07lr_mult\x18\x03 \x01(\x02:\x01\x31\x12\x15\n\ndecay_mult\x18\x04 \x01(\x02:\x01\x31\"*\n\x0c\x44imCheckMode\x12\n\n\x06STRICT\x10\x00\x12\x0e\n\nPERMISSIVE\x10\x01\"\x95%\n\x0eLayerParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0e\n\x06\x62ottom\x18\x03 \x03(\t\x12\x0b\n\x03top\x18\x04 \x03(\t\x12\x1b\n\x05phase\x18\n \x01(\x0e\x32\x0c.caffe.Phase\x12\x13\n\x0bloss_weight\x18\x05 \x03(\x02\x12\x1f\n\x05param\x18\x06 \x03(\x0b\x32\x10.caffe.ParamSpec\x12\x1f\n\x05\x62lobs\x18\x07 \x03(\x0b\x32\x10.caffe.BlobProto\x12\x16\n\x0epropagate_down\x18\x0b \x03(\x08\x12$\n\x07include\x18\x08 \x03(\x0b\x32\x13.caffe.NetStateRule\x12$\n\x07\x65xclude\x18\t \x03(\x0b\x32\x13.caffe.NetStateRule\x12\x37\n\x0ftransform_param\x18\x64 \x01(\x0b\x32\x1e.caffe.TransformationParameter\x12(\n\nloss_param\x18\x65 \x01(\x0b\x32\x14.caffe.LossParameter\x12<\n\x14\x64\x65tection_loss_param\x18\xc8\x01 \x01(\x0b\x32\x1d.caffe.DetectionLossParameter\x12<\n\x14\x65val_detection_param\x18\xc9\x01 \x01(\x0b\x32\x1d.caffe.EvalDetectionParameter\x12\x36\n\x11region_loss_param\x18\xca\x01 \x01(\x0b\x32\x1a.caffe.RegionLossParameter\x12+\n\x0breorg_param\x18\xcb\x01 \x01(\x0b\x32\x15.caffe.ReorgParameter\x12\x30\n\x0e\x61\x63\x63uracy_param\x18\x66 \x01(\x0b\x32\x18.caffe.AccuracyParameter\x12,\n\x0c\x61rgmax_param\x18g \x01(\x0b\x32\x16.caffe.ArgMaxParameter\x12\x34\n\x10\x62\x61tch_norm_param\x18\x8b\x01 \x01(\x0b\x32\x19.caffe.BatchNormParameter\x12)\n\nbias_param\x18\x8d\x01 \x01(\x0b\x32\x14.caffe.BiasParameter\x12,\n\x0c\x63oncat_param\x18h \x01(\x0b\x32\x16.caffe.ConcatParameter\x12?\n\x16\x63ontrastive_loss_param\x18i \x01(\x0b\x32\x1f.caffe.ContrastiveLossParameter\x12\x36\n\x11\x63onvolution_param\x18j \x01(\x0b\x32\x1b.caffe.ConvolutionParameter\x12(\n\ndata_param\x18k \x01(\x0b\x32\x14.caffe.DataParameter\x12.\n\rdropout_param\x18l \x01(\x0b\x32\x17.caffe.DropoutParameter\x12\x33\n\x10\x64ummy_data_param\x18m \x01(\x0b\x32\x19.caffe.DummyDataParameter\x12.\n\reltwise_param\x18n \x01(\x0b\x32\x17.caffe.EltwiseParameter\x12\'\n\telu_param\x18\x8c\x01 \x01(\x0b\x32\x13.caffe.ELUParameter\x12+\n\x0b\x65mbed_param\x18\x89\x01 \x01(\x0b\x32\x15.caffe.EmbedParameter\x12&\n\texp_param\x18o \x01(\x0b\x32\x13.caffe.ExpParameter\x12/\n\rflatten_param\x18\x87\x01 \x01(\x0b\x32\x17.caffe.FlattenParameter\x12\x31\n\x0fhdf5_data_param\x18p \x01(\x0b\x32\x18.caffe.HDF5DataParameter\x12\x35\n\x11hdf5_output_param\x18q \x01(\x0b\x32\x1a.caffe.HDF5OutputParameter\x12\x33\n\x10hinge_loss_param\x18r \x01(\x0b\x32\x19.caffe.HingeLossParameter\x12\x33\n\x10image_data_param\x18s \x01(\x0b\x32\x19.caffe.ImageDataParameter\x12\x39\n\x13infogain_loss_param\x18t \x01(\x0b\x32\x1c.caffe.InfogainLossParameter\x12\x39\n\x13inner_product_param\x18u \x01(\x0b\x32\x1c.caffe.InnerProductParameter\x12+\n\x0binput_param\x18\x8f\x01 \x01(\x0b\x32\x15.caffe.InputParameter\x12\'\n\tlog_param\x18\x86\x01 \x01(\x0b\x32\x13.caffe.LogParameter\x12&\n\tlrn_param\x18v \x01(\x0b\x32\x13.caffe.LRNParameter\x12\x35\n\x11memory_data_param\x18w \x01(\x0b\x32\x1a.caffe.MemoryDataParameter\x12&\n\tmvn_param\x18x \x01(\x0b\x32\x13.caffe.MVNParameter\x12.\n\rpooling_param\x18y \x01(\x0b\x32\x17.caffe.PoolingParameter\x12*\n\x0bpower_param\x18z \x01(\x0b\x32\x15.caffe.PowerParameter\x12+\n\x0bprelu_param\x18\x83\x01 \x01(\x0b\x32\x15.caffe.PReLUParameter\x12-\n\x0cpython_param\x18\x82\x01 \x01(\x0b\x32\x16.caffe.PythonParameter\x12\x33\n\x0frecurrent_param\x18\x92\x01 \x01(\x0b\x32\x19.caffe.RecurrentParameter\x12\x33\n\x0freduction_param\x18\x88\x01 \x01(\x0b\x32\x19.caffe.ReductionParameter\x12(\n\nrelu_param\x18{ \x01(\x0b\x32\x14.caffe.ReLUParameter\x12/\n\rreshape_param\x18\x85\x01 \x01(\x0b\x32\x17.caffe.ReshapeParameter\x12\x38\n\x11roi_pooling_param\x18\xd7\xc7\xf8\x03 \x01(\x0b\x32\x1a.caffe.ROIPoolingParameter\x12+\n\x0bscale_param\x18\x8e\x01 \x01(\x0b\x32\x15.caffe.ScaleParameter\x12.\n\rsigmoid_param\x18| \x01(\x0b\x32\x17.caffe.SigmoidParameter\x12=\n\x14smooth_l1_loss_param\x18\xd8\xc7\xf8\x03 \x01(\x0b\x32\x1c.caffe.SmoothL1LossParameter\x12.\n\rsoftmax_param\x18} \x01(\x0b\x32\x17.caffe.SoftmaxParameter\x12\'\n\tspp_param\x18\x84\x01 \x01(\x0b\x32\x13.caffe.SPPParameter\x12*\n\x0bslice_param\x18~ \x01(\x0b\x32\x15.caffe.SliceParameter\x12(\n\ntanh_param\x18\x7f \x01(\x0b\x32\x14.caffe.TanHParameter\x12\x33\n\x0fthreshold_param\x18\x80\x01 \x01(\x0b\x32\x19.caffe.ThresholdParameter\x12)\n\ntile_param\x18\x8a\x01 \x01(\x0b\x32\x14.caffe.TileParameter\x12\x36\n\x11window_data_param\x18\x81\x01 \x01(\x0b\x32\x1a.caffe.WindowDataParameter\x12\x35\n\x08st_param\x18\x94\x01 \x01(\x0b\x32\".caffe.SpatialTransformerParameter\x12.\n\rst_loss_param\x18\x91\x01 \x01(\x0b\x32\x16.caffe.STLossParameter\x12\'\n\trpn_param\x18\x96\x01 \x01(\x0b\x32\x13.caffe.RPNParameter\x12\x34\n\x10\x66ocal_loss_param\x18\x9b\x01 \x01(\x0b\x32\x19.caffe.FocalLossParameter\x12\x32\n\x0f\x61sdn_data_param\x18\x9f\x01 \x01(\x0b\x32\x18.caffe.AsdnDataParameter\x12%\n\x08\x62n_param\x18\xa0\x01 \x01(\x0b\x32\x12.caffe.BNParameter\x12\x34\n\x10mtcnn_data_param\x18\xa1\x01 \x01(\x0b\x32\x19.caffe.MTCNNDataParameter\x12-\n\x0cinterp_param\x18\xa2\x01 \x01(\x0b\x32\x16.caffe.InterpParameter\x12:\n\x13psroi_pooling_param\x18\xa3\x01 \x01(\x0b\x32\x1c.caffe.PSROIPoolingParameter\x12<\n\x14\x61nnotated_data_param\x18\xa4\x01 \x01(\x0b\x32\x1d.caffe.AnnotatedDataParameter\x12\x32\n\x0fprior_box_param\x18\xa5\x01 \x01(\x0b\x32\x18.caffe.PriorBoxParameter\x12)\n\ncrop_param\x18\xa7\x01 \x01(\x0b\x32\x14.caffe.CropParameter\x12\x44\n\x18\x64\x65tection_evaluate_param\x18\xa8\x01 \x01(\x0b\x32!.caffe.DetectionEvaluateParameter\x12@\n\x16\x64\x65tection_output_param\x18\xa9\x01 \x01(\x0b\x32\x1f.caffe.DetectionOutputParameter\x12:\n\x13multibox_loss_param\x18\xab\x01 \x01(\x0b\x32\x1c.caffe.MultiBoxLossParameter\x12/\n\rpermute_param\x18\xac\x01 \x01(\x0b\x32\x17.caffe.PermuteParameter\x12\x34\n\x10video_data_param\x18\xad\x01 \x01(\x0b\x32\x19.caffe.VideoDataParameter\x12G\n\x1amargin_inner_product_param\x18\xae\x01 \x01(\x0b\x32\".caffe.MarginInnerProductParameter\x12\x36\n\x11\x63\x65nter_loss_param\x18\xaf\x01 \x01(\x0b\x32\x1a.caffe.CenterLossParameter\x12L\n\x1c\x64\x65\x66ormable_convolution_param\x18\xb0\x01 \x01(\x0b\x32%.caffe.DeformableConvolutionParameter\x12\x43\n\x18label_specific_add_param\x18\xb1\x01 \x01(\x0b\x32 .caffe.LabelSpecificAddParameter\x12X\n#additive_margin_inner_product_param\x18\xb2\x01 \x01(\x0b\x32*.caffe.AdditiveMarginInnerProductParameter\x12\x35\n\x11\x63osin_add_m_param\x18\xb3\x01 \x01(\x0b\x32\x19.caffe.CosinAddmParameter\x12\x35\n\x11\x63osin_mul_m_param\x18\xb4\x01 \x01(\x0b\x32\x19.caffe.CosinMulmParameter\x12:\n\x13\x63hannel_scale_param\x18\xb5\x01 \x01(\x0b\x32\x1c.caffe.ChannelScaleParameter\x12)\n\nflip_param\x18\xb6\x01 \x01(\x0b\x32\x14.caffe.FlipParameter\x12\x38\n\x12triplet_loss_param\x18\xb7\x01 \x01(\x0b\x32\x1b.caffe.TripletLossParameter\x12G\n\x1a\x63oupled_cluster_loss_param\x18\xb8\x01 \x01(\x0b\x32\".caffe.CoupledClusterLossParameter\x12\x43\n\x1ageneral_triplet_loss_param\x18\xb9\x01 \x01(\x0b\x32\x1e.caffe.GeneralTripletParameter\x12\x32\n\x0froi_align_param\x18\xba\x01 \x01(\x0b\x32\x18.caffe.ROIAlignParameter\x12\x32\n\x0eupsample_param\x18\xa3\x8d\x06 \x01(\x0b\x32\x18.caffe.UpsampleParameter\x12.\n\x0cmatmul_param\x18\xa5\x8d\x06 \x01(\x0b\x32\x16.caffe.MatMulParameter\x12\x39\n\x12pass_through_param\x18\xa4\x8d\x06 \x01(\x0b\x32\x1b.caffe.PassThroughParameter\x12/\n\nnorm_param\x18\xa1\x8d\x06 \x01(\x0b\x32\x19.caffe.NormalizeParameter\"\xa3\x01\n\x11UpsampleParameter\x12\x10\n\x05scale\x18\x01 \x01(\r:\x01\x32\x12\x0f\n\x07scale_h\x18\x02 \x01(\r\x12\x0f\n\x07scale_w\x18\x03 \x01(\r\x12\x18\n\tpad_out_h\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x18\n\tpad_out_w\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x12\n\nupsample_h\x18\x06 \x01(\r\x12\x12\n\nupsample_w\x18\x07 \x01(\r\">\n\x0fMatMulParameter\x12\r\n\x05\x64im_1\x18\x01 \x01(\r\x12\r\n\x05\x64im_2\x18\x02 \x01(\r\x12\r\n\x05\x64im_3\x18\x03 \x01(\r\"^\n\x14PassThroughParameter\x12\x15\n\nnum_output\x18\x01 \x01(\r:\x01\x30\x12\x17\n\x0c\x62lock_height\x18\x02 \x01(\r:\x01\x30\x12\x16\n\x0b\x62lock_width\x18\x03 \x01(\r:\x01\x30\"\xa5\x01\n\x12NormalizeParameter\x12\x1c\n\x0e\x61\x63ross_spatial\x18\x01 \x01(\x08:\x04true\x12,\n\x0cscale_filler\x18\x02 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x1c\n\x0e\x63hannel_shared\x18\x03 \x01(\x08:\x04true\x12\x12\n\x03\x65ps\x18\x04 \x01(\x02:\x05\x31\x65-10\x12\x11\n\x06sqrt_a\x18\x05 \x01(\x02:\x01\x31\"\x95\x01\n\x16\x41nnotatedDataParameter\x12*\n\rbatch_sampler\x18\x01 \x03(\x0b\x32\x13.caffe.BatchSampler\x12\x16\n\x0elabel_map_file\x18\x02 \x01(\t\x12\x37\n\tanno_type\x18\x03 \x01(\x0e\x32$.caffe.AnnotatedDatum.AnnotationType\"\xab\x01\n\x11\x41sdnDataParameter\x12\x16\n\ncount_drop\x18\x01 \x01(\x05:\x02\x31\x35\x12\x19\n\rpermute_count\x18\x02 \x01(\x05:\x02\x32\x30\x12\x19\n\x0e\x63ount_drop_neg\x18\x03 \x01(\x05:\x01\x30\x12\x16\n\x08\x63hannels\x18\x04 \x01(\x05:\x04\x31\x30\x32\x34\x12\x14\n\titer_size\x18\x05 \x01(\x05:\x01\x32\x12\x1a\n\x0fmaintain_before\x18\x06 \x01(\x05:\x01\x31\"\x80\x02\n\x12MTCNNDataParameter\x12\x17\n\taugmented\x18\x01 \x01(\x08:\x04true\x12\x12\n\x04\x66lip\x18\x02 \x01(\x08:\x04true\x12\x18\n\x0cnum_positive\x18\x03 \x01(\x05:\x02-1\x12\x18\n\x0cnum_negitive\x18\x04 \x01(\x05:\x02-1\x12\x14\n\x08num_part\x18\x05 \x01(\x05:\x02-1\x12\x17\n\x0cresize_width\x18\x06 \x01(\r:\x01\x30\x12\x18\n\rresize_height\x18\x07 \x01(\r:\x01\x30\x12\x1f\n\x12min_negitive_scale\x18\x08 \x01(\x02:\x03\x30.5\x12\x1f\n\x12max_negitive_scale\x18\t \x01(\x02:\x03\x31.5\"\x90\x01\n\x0fInterpParameter\x12\x11\n\x06height\x18\x01 \x01(\x05:\x01\x30\x12\x10\n\x05width\x18\x02 \x01(\x05:\x01\x30\x12\x16\n\x0bzoom_factor\x18\x03 \x01(\x05:\x01\x31\x12\x18\n\rshrink_factor\x18\x04 \x01(\x05:\x01\x31\x12\x12\n\x07pad_beg\x18\x05 \x01(\x05:\x01\x30\x12\x12\n\x07pad_end\x18\x06 \x01(\x05:\x01\x30\"V\n\x15PSROIPoolingParameter\x12\x15\n\rspatial_scale\x18\x01 \x02(\x02\x12\x12\n\noutput_dim\x18\x02 \x02(\x05\x12\x12\n\ngroup_size\x18\x03 \x02(\x05\"E\n\rFlipParameter\x12\x18\n\nflip_width\x18\x01 \x01(\x08:\x04true\x12\x1a\n\x0b\x66lip_height\x18\x02 \x01(\x08:\x05\x66\x61lse\"\x8b\x02\n\x0b\x42NParameter\x12,\n\x0cslope_filler\x18\x01 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x02 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x15\n\x08momentum\x18\x03 \x01(\x02:\x03\x30.9\x12\x12\n\x03\x65ps\x18\x04 \x01(\x02:\x05\x31\x65-05\x12\x15\n\x06\x66rozen\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x32\n\x06\x65ngine\x18\x06 \x01(\x0e\x32\x19.caffe.BNParameter.Engine:\x07\x44\x45\x46\x41ULT\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"\xa2\x01\n\x12\x46ocalLossParameter\x12\x34\n\x04type\x18\x01 \x01(\x0e\x32\x1e.caffe.FocalLossParameter.Type:\x06ORIGIN\x12\x10\n\x05gamma\x18\x02 \x01(\x02:\x01\x32\x12\x13\n\x05\x61lpha\x18\x03 \x01(\x02:\x04\x30.25\x12\x0f\n\x04\x62\x65ta\x18\x04 \x01(\x02:\x01\x31\"\x1e\n\x04Type\x12\n\n\x06ORIGIN\x10\x00\x12\n\n\x06LINEAR\x10\x01\"\xca\x03\n\x17TransformationParameter\x12\x10\n\x05scale\x18\x01 \x01(\x02:\x01\x31\x12\x15\n\x06mirror\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x14\n\tcrop_size\x18\x03 \x01(\r:\x01\x30\x12\x11\n\x06\x63rop_h\x18\x0b \x01(\r:\x01\x30\x12\x11\n\x06\x63rop_w\x18\x0c \x01(\r:\x01\x30\x12\x11\n\tmean_file\x18\x04 \x01(\t\x12\x12\n\nmean_value\x18\x05 \x03(\x02\x12\x1a\n\x0b\x66orce_color\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x19\n\nforce_gray\x18\x07 \x01(\x08:\x05\x66\x61lse\x12,\n\x0cresize_param\x18\x08 \x01(\x0b\x32\x16.caffe.ResizeParameter\x12*\n\x0bnoise_param\x18\t \x01(\x0b\x32\x15.caffe.NoiseParameter\x12\x31\n\rdistort_param\x18\r \x01(\x0b\x32\x1a.caffe.DistortionParameter\x12/\n\x0c\x65xpand_param\x18\x0e \x01(\x0b\x32\x19.caffe.ExpansionParameter\x12.\n\x0f\x65mit_constraint\x18\n \x01(\x0b\x32\x15.caffe.EmitConstraint\"\x90\x04\n\x0fResizeParameter\x12\x0f\n\x04prob\x18\x01 \x01(\x02:\x01\x31\x12=\n\x0bresize_mode\x18\x02 \x01(\x0e\x32\".caffe.ResizeParameter.Resize_mode:\x04WARP\x12\x11\n\x06height\x18\x03 \x01(\r:\x01\x30\x12\x10\n\x05width\x18\x04 \x01(\r:\x01\x30\x12\x17\n\x0cheight_scale\x18\x08 \x01(\r:\x01\x30\x12\x16\n\x0bwidth_scale\x18\t \x01(\r:\x01\x30\x12;\n\x08pad_mode\x18\x05 \x01(\x0e\x32\x1f.caffe.ResizeParameter.Pad_mode:\x08\x43ONSTANT\x12\x11\n\tpad_value\x18\x06 \x03(\x02\x12\x37\n\x0binterp_mode\x18\x07 \x03(\x0e\x32\".caffe.ResizeParameter.Interp_mode\"G\n\x0bResize_mode\x12\x08\n\x04WARP\x10\x01\x12\x12\n\x0e\x46IT_SMALL_SIZE\x10\x02\x12\x1a\n\x16\x46IT_LARGE_SIZE_AND_PAD\x10\x03\":\n\x08Pad_mode\x12\x0c\n\x08\x43ONSTANT\x10\x01\x12\x0c\n\x08MIRRORED\x10\x02\x12\x12\n\x0eREPEAT_NEAREST\x10\x03\"I\n\x0bInterp_mode\x12\n\n\x06LINEAR\x10\x01\x12\x08\n\x04\x41REA\x10\x02\x12\x0b\n\x07NEAREST\x10\x03\x12\t\n\x05\x43UBIC\x10\x04\x12\x0c\n\x08LANCZOS4\x10\x05\"9\n\x13SaltPepperParameter\x12\x13\n\x08\x66raction\x18\x01 \x01(\x02:\x01\x30\x12\r\n\x05value\x18\x02 \x03(\x02\"\xee\x02\n\x0eNoiseParameter\x12\x0f\n\x04prob\x18\x01 \x01(\x02:\x01\x30\x12\x16\n\x07hist_eq\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x16\n\x07inverse\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x19\n\ndecolorize\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x19\n\ngauss_blur\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x10\n\x04jpeg\x18\x06 \x01(\x02:\x02-1\x12\x18\n\tposterize\x18\x07 \x01(\x08:\x05\x66\x61lse\x12\x14\n\x05\x65rode\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x19\n\nsaltpepper\x18\t \x01(\x08:\x05\x66\x61lse\x12\x34\n\x10saltpepper_param\x18\n \x01(\x0b\x32\x1a.caffe.SaltPepperParameter\x12\x14\n\x05\x63lahe\x18\x0b \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x0e\x63onvert_to_hsv\x18\x0c \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x0e\x63onvert_to_lab\x18\r \x01(\x08:\x05\x66\x61lse\"\xbd\x02\n\x13\x44istortionParameter\x12\x1a\n\x0f\x62rightness_prob\x18\x01 \x01(\x02:\x01\x30\x12\x1b\n\x10\x62rightness_delta\x18\x02 \x01(\x02:\x01\x30\x12\x18\n\rcontrast_prob\x18\x03 \x01(\x02:\x01\x30\x12\x19\n\x0e\x63ontrast_lower\x18\x04 \x01(\x02:\x01\x30\x12\x19\n\x0e\x63ontrast_upper\x18\x05 \x01(\x02:\x01\x30\x12\x13\n\x08hue_prob\x18\x06 \x01(\x02:\x01\x30\x12\x14\n\thue_delta\x18\x07 \x01(\x02:\x01\x30\x12\x1a\n\x0fsaturation_prob\x18\x08 \x01(\x02:\x01\x30\x12\x1b\n\x10saturation_lower\x18\t \x01(\x02:\x01\x30\x12\x1b\n\x10saturation_upper\x18\n \x01(\x02:\x01\x30\x12\x1c\n\x11random_order_prob\x18\x0b \x01(\x02:\x01\x30\"B\n\x12\x45xpansionParameter\x12\x0f\n\x04prob\x18\x01 \x01(\x02:\x01\x31\x12\x1b\n\x10max_expand_ratio\x18\x02 \x01(\x02:\x01\x31\"\xc2\x01\n\rLossParameter\x12\x14\n\x0cignore_label\x18\x01 \x01(\x05\x12\x44\n\rnormalization\x18\x03 \x01(\x0e\x32&.caffe.LossParameter.NormalizationMode:\x05VALID\x12\x11\n\tnormalize\x18\x02 \x01(\x08\"B\n\x11NormalizationMode\x12\x08\n\x04\x46ULL\x10\x00\x12\t\n\x05VALID\x10\x01\x12\x0e\n\nBATCH_SIZE\x10\x02\x12\x08\n\x04NONE\x10\x03\"L\n\x11\x41\x63\x63uracyParameter\x12\x10\n\x05top_k\x18\x01 \x01(\r:\x01\x31\x12\x0f\n\x04\x61xis\x18\x02 \x01(\x05:\x01\x31\x12\x14\n\x0cignore_label\x18\x03 \x01(\x05\"M\n\x0f\x41rgMaxParameter\x12\x1a\n\x0bout_max_val\x18\x01 \x01(\x08:\x05\x66\x61lse\x12\x10\n\x05top_k\x18\x02 \x01(\r:\x01\x31\x12\x0c\n\x04\x61xis\x18\x03 \x01(\x05\"9\n\x0f\x43oncatParameter\x12\x0f\n\x04\x61xis\x18\x02 \x01(\x05:\x01\x31\x12\x15\n\nconcat_dim\x18\x01 \x01(\r:\x01\x31\"j\n\x12\x42\x61tchNormParameter\x12\x18\n\x10use_global_stats\x18\x01 \x01(\x08\x12&\n\x17moving_average_fraction\x18\x02 \x01(\x02:\x05\x30.999\x12\x12\n\x03\x65ps\x18\x03 \x01(\x02:\x05\x31\x65-05\"]\n\rBiasParameter\x12\x0f\n\x04\x61xis\x18\x01 \x01(\x05:\x01\x31\x12\x13\n\x08num_axes\x18\x02 \x01(\x05:\x01\x31\x12&\n\x06\x66iller\x18\x03 \x01(\x0b\x32\x16.caffe.FillerParameter\"L\n\x18\x43ontrastiveLossParameter\x12\x11\n\x06margin\x18\x01 \x01(\x02:\x01\x31\x12\x1d\n\x0elegacy_version\x18\x02 \x01(\x08:\x05\x66\x61lse\"\xec\x01\n\x16\x44\x65tectionLossParameter\x12\x0f\n\x04side\x18\x01 \x01(\r:\x01\x37\x12\x15\n\tnum_class\x18\x02 \x01(\r:\x02\x32\x30\x12\x15\n\nnum_object\x18\x03 \x01(\r:\x01\x32\x12\x17\n\x0cobject_scale\x18\x04 \x01(\x02:\x01\x31\x12\x1b\n\x0enoobject_scale\x18\x05 \x01(\x02:\x03\x30.5\x12\x16\n\x0b\x63lass_scale\x18\x06 \x01(\x02:\x01\x31\x12\x16\n\x0b\x63oord_scale\x18\x07 \x01(\x02:\x01\x35\x12\x12\n\x04sqrt\x18\x08 \x01(\x08:\x04true\x12\x19\n\nconstriant\x18\t \x01(\x08:\x05\x66\x61lse\"\x91\x03\n\x13RegionLossParameter\x12\x10\n\x04side\x18\x01 \x01(\r:\x02\x31\x33\x12\x15\n\tnum_class\x18\x02 \x01(\r:\x02\x32\x30\x12\x15\n\nbias_match\x18\x03 \x01(\r:\x01\x31\x12\x11\n\x06\x63oords\x18\x04 \x01(\r:\x01\x34\x12\x0e\n\x03num\x18\x05 \x01(\r:\x01\x35\x12\x12\n\x07softmax\x18\x06 \x01(\r:\x01\x31\x12\x13\n\x06jitter\x18\x07 \x01(\x02:\x03\x30.2\x12\x12\n\x07rescore\x18\x08 \x01(\r:\x01\x31\x12\x17\n\x0cobject_scale\x18\t \x01(\x02:\x01\x31\x12\x16\n\x0b\x63lass_scale\x18\n \x01(\x02:\x01\x31\x12\x1b\n\x0enoobject_scale\x18\x0b \x01(\x02:\x03\x30.5\x12\x16\n\x0b\x63oord_scale\x18\x0c \x01(\x02:\x01\x35\x12\x13\n\x08\x61\x62solute\x18\r \x01(\r:\x01\x31\x12\x13\n\x06thresh\x18\x0e \x01(\x02:\x03\x30.2\x12\x11\n\x06random\x18\x0f \x01(\r:\x01\x31\x12\x0e\n\x06\x62iases\x18\x10 \x03(\x02\x12\x14\n\x0csoftmax_tree\x18\x11 \x01(\t\x12\x11\n\tclass_map\x18\x12 \x01(\t\"8\n\x0eReorgParameter\x12\x0e\n\x06stride\x18\x01 \x01(\r\x12\x16\n\x07reverse\x18\x02 \x01(\x08:\x05\x66\x61lse\"\xb3\x02\n\x16\x45valDetectionParameter\x12\x0f\n\x04side\x18\x01 \x01(\r:\x01\x37\x12\x15\n\tnum_class\x18\x02 \x01(\r:\x02\x32\x30\x12\x15\n\nnum_object\x18\x03 \x01(\r:\x01\x32\x12\x16\n\tthreshold\x18\x04 \x01(\x02:\x03\x30.5\x12\x12\n\x04sqrt\x18\x05 \x01(\x08:\x04true\x12\x18\n\nconstriant\x18\x06 \x01(\x08:\x04true\x12\x45\n\nscore_type\x18\x07 \x01(\x0e\x32\'.caffe.EvalDetectionParameter.ScoreType:\x08MULTIPLY\x12\x0f\n\x03nms\x18\x08 \x01(\x02:\x02-1\x12\x0e\n\x06\x62iases\x18\t \x03(\x02\",\n\tScoreType\x12\x07\n\x03OBJ\x10\x00\x12\x08\n\x04PROB\x10\x01\x12\x0c\n\x08MULTIPLY\x10\x02\"\xfc\x03\n\x14\x43onvolutionParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12\x17\n\tbias_term\x18\x02 \x01(\x08:\x04true\x12\x0b\n\x03pad\x18\x03 \x03(\r\x12\x13\n\x0bkernel_size\x18\x04 \x03(\r\x12\x0e\n\x06stride\x18\x06 \x03(\r\x12\x10\n\x08\x64ilation\x18\x12 \x03(\r\x12\x10\n\x05pad_h\x18\t \x01(\r:\x01\x30\x12\x10\n\x05pad_w\x18\n \x01(\r:\x01\x30\x12\x10\n\x08kernel_h\x18\x0b \x01(\r\x12\x10\n\x08kernel_w\x18\x0c \x01(\r\x12\x10\n\x08stride_h\x18\r \x01(\r\x12\x10\n\x08stride_w\x18\x0e \x01(\r\x12\x10\n\x05group\x18\x05 \x01(\r:\x01\x31\x12-\n\rweight_filler\x18\x07 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x08 \x01(\x0b\x32\x16.caffe.FillerParameter\x12;\n\x06\x65ngine\x18\x0f \x01(\x0e\x32\".caffe.ConvolutionParameter.Engine:\x07\x44\x45\x46\x41ULT\x12\x0f\n\x04\x61xis\x18\x10 \x01(\x05:\x01\x31\x12\x1e\n\x0f\x66orce_nd_im2col\x18\x11 \x01(\x08:\x05\x66\x61lse\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"0\n\rCropParameter\x12\x0f\n\x04\x61xis\x18\x01 \x01(\x05:\x01\x32\x12\x0e\n\x06offset\x18\x02 \x03(\r\"\xb2\x02\n\rDataParameter\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x12\n\nbatch_size\x18\x04 \x01(\r\x12\x14\n\trand_skip\x18\x07 \x01(\r:\x01\x30\x12\x31\n\x07\x62\x61\x63kend\x18\x08 \x01(\x0e\x32\x17.caffe.DataParameter.DB:\x07LEVELDB\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x31\x12\x11\n\tmean_file\x18\x03 \x01(\t\x12\x14\n\tcrop_size\x18\x05 \x01(\r:\x01\x30\x12\x15\n\x06mirror\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\"\n\x13\x66orce_encoded_color\x18\t \x01(\x08:\x05\x66\x61lse\x12\x13\n\x08prefetch\x18\n \x01(\r:\x01\x34\x12\x0c\n\x04side\x18\x0b \x03(\r\"\x1b\n\x02\x44\x42\x12\x0b\n\x07LEVELDB\x10\x00\x12\x08\n\x04LMDB\x10\x01\"\xdc\x01\n\x1a\x44\x65tectionEvaluateParameter\x12\x13\n\x0bnum_classes\x18\x01 \x01(\r\x12\x1e\n\x13\x62\x61\x63kground_label_id\x18\x02 \x01(\r:\x01\x30\x12\x1e\n\x11overlap_threshold\x18\x03 \x01(\x02:\x03\x30.5\x12#\n\x15\x65valuate_difficult_gt\x18\x04 \x01(\x08:\x04true\x12\x16\n\x0ename_size_file\x18\x05 \x01(\t\x12,\n\x0cresize_param\x18\x06 \x01(\x0b\x32\x16.caffe.ResizeParameter\"[\n\x1eNonMaximumSuppressionParameter\x12\x1a\n\rnms_threshold\x18\x01 \x01(\x02:\x03\x30.3\x12\r\n\x05top_k\x18\x02 \x01(\x05\x12\x0e\n\x03\x65ta\x18\x03 \x01(\x02:\x01\x31\"\xd8\x01\n\x13SaveOutputParameter\x12\x18\n\x10output_directory\x18\x01 \x01(\t\x12\x1a\n\x12output_name_prefix\x18\x02 \x01(\t\x12\x15\n\routput_format\x18\x03 \x01(\t\x12\x16\n\x0elabel_map_file\x18\x04 \x01(\t\x12\x16\n\x0ename_size_file\x18\x05 \x01(\t\x12\x16\n\x0enum_test_image\x18\x06 \x01(\r\x12,\n\x0cresize_param\x18\x07 \x01(\x0b\x32\x16.caffe.ResizeParameter\"\xc7\x03\n\x18\x44\x65tectionOutputParameter\x12\x13\n\x0bnum_classes\x18\x01 \x01(\r\x12\x1c\n\x0eshare_location\x18\x02 \x01(\x08:\x04true\x12\x1e\n\x13\x62\x61\x63kground_label_id\x18\x03 \x01(\x05:\x01\x30\x12\x38\n\tnms_param\x18\x04 \x01(\x0b\x32%.caffe.NonMaximumSuppressionParameter\x12\x35\n\x11save_output_param\x18\x05 \x01(\x0b\x32\x1a.caffe.SaveOutputParameter\x12<\n\tcode_type\x18\x06 \x01(\x0e\x32!.caffe.PriorBoxParameter.CodeType:\x06\x43ORNER\x12)\n\x1avariance_encoded_in_target\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x16\n\nkeep_top_k\x18\x07 \x01(\x05:\x02-1\x12\x1c\n\x14\x63onfidence_threshold\x18\t \x01(\x02\x12\x18\n\tvisualize\x18\n \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x13visualize_threshold\x18\x0b \x01(\x02\x12\x11\n\tsave_file\x18\x0c \x01(\t\"I\n\x10\x44ropoutParameter\x12\x1a\n\rdropout_ratio\x18\x01 \x01(\x02:\x03\x30.5\x12\x19\n\x0bscale_train\x18\x02 \x01(\x08:\x04true\"\xa0\x01\n\x12\x44ummyDataParameter\x12+\n\x0b\x64\x61ta_filler\x18\x01 \x03(\x0b\x32\x16.caffe.FillerParameter\x12\x1f\n\x05shape\x18\x06 \x03(\x0b\x32\x10.caffe.BlobShape\x12\x0b\n\x03num\x18\x02 \x03(\r\x12\x10\n\x08\x63hannels\x18\x03 \x03(\r\x12\x0e\n\x06height\x18\x04 \x03(\r\x12\r\n\x05width\x18\x05 \x03(\r\"\xa5\x01\n\x10\x45ltwiseParameter\x12\x39\n\toperation\x18\x01 \x01(\x0e\x32!.caffe.EltwiseParameter.EltwiseOp:\x03SUM\x12\r\n\x05\x63oeff\x18\x02 \x03(\x02\x12\x1e\n\x10stable_prod_grad\x18\x03 \x01(\x08:\x04true\"\'\n\tEltwiseOp\x12\x08\n\x04PROD\x10\x00\x12\x07\n\x03SUM\x10\x01\x12\x07\n\x03MAX\x10\x02\" \n\x0c\x45LUParameter\x12\x10\n\x05\x61lpha\x18\x01 \x01(\x02:\x01\x31\"\xac\x01\n\x0e\x45mbedParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12\x11\n\tinput_dim\x18\x02 \x01(\r\x12\x17\n\tbias_term\x18\x03 \x01(\x08:\x04true\x12-\n\rweight_filler\x18\x04 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x05 \x01(\x0b\x32\x16.caffe.FillerParameter\"D\n\x0c\x45xpParameter\x12\x10\n\x04\x62\x61se\x18\x01 \x01(\x02:\x02-1\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x31\x12\x10\n\x05shift\x18\x03 \x01(\x02:\x01\x30\"9\n\x10\x46lattenParameter\x12\x0f\n\x04\x61xis\x18\x01 \x01(\x05:\x01\x31\x12\x14\n\x08\x65nd_axis\x18\x02 \x01(\x05:\x02-1\"O\n\x11HDF5DataParameter\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x12\n\nbatch_size\x18\x02 \x01(\r\x12\x16\n\x07shuffle\x18\x03 \x01(\x08:\x05\x66\x61lse\"(\n\x13HDF5OutputParameter\x12\x11\n\tfile_name\x18\x01 \x01(\t\"^\n\x12HingeLossParameter\x12\x30\n\x04norm\x18\x01 \x01(\x0e\x32\x1e.caffe.HingeLossParameter.Norm:\x02L1\"\x16\n\x04Norm\x12\x06\n\x02L1\x10\x01\x12\x06\n\x02L2\x10\x02\"\x97\x02\n\x12ImageDataParameter\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x15\n\nbatch_size\x18\x04 \x01(\r:\x01\x31\x12\x14\n\trand_skip\x18\x07 \x01(\r:\x01\x30\x12\x16\n\x07shuffle\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x15\n\nnew_height\x18\t \x01(\r:\x01\x30\x12\x14\n\tnew_width\x18\n \x01(\r:\x01\x30\x12\x16\n\x08is_color\x18\x0b \x01(\x08:\x04true\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x31\x12\x11\n\tmean_file\x18\x03 \x01(\t\x12\x14\n\tcrop_size\x18\x05 \x01(\r:\x01\x30\x12\x15\n\x06mirror\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x15\n\x0broot_folder\x18\x0c \x01(\t:\x00\"\'\n\x15InfogainLossParameter\x12\x0e\n\x06source\x18\x01 \x01(\t\"\xe5\x01\n\x15InnerProductParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12\x17\n\tbias_term\x18\x02 \x01(\x08:\x04true\x12-\n\rweight_filler\x18\x03 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x04 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x0f\n\x04\x61xis\x18\x05 \x01(\x05:\x01\x31\x12\x18\n\ttranspose\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x18\n\tnormalize\x18\x07 \x01(\x08:\x05\x66\x61lse\"1\n\x0eInputParameter\x12\x1f\n\x05shape\x18\x01 \x03(\x0b\x32\x10.caffe.BlobShape\"D\n\x0cLogParameter\x12\x10\n\x04\x62\x61se\x18\x01 \x01(\x02:\x02-1\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x31\x12\x10\n\x05shift\x18\x03 \x01(\x02:\x01\x30\"\xb8\x02\n\x0cLRNParameter\x12\x15\n\nlocal_size\x18\x01 \x01(\r:\x01\x35\x12\x10\n\x05\x61lpha\x18\x02 \x01(\x02:\x01\x31\x12\x12\n\x04\x62\x65ta\x18\x03 \x01(\x02:\x04\x30.75\x12\x44\n\x0bnorm_region\x18\x04 \x01(\x0e\x32\x1e.caffe.LRNParameter.NormRegion:\x0f\x41\x43ROSS_CHANNELS\x12\x0c\n\x01k\x18\x05 \x01(\x02:\x01\x31\x12\x33\n\x06\x65ngine\x18\x06 \x01(\x0e\x32\x1a.caffe.LRNParameter.Engine:\x07\x44\x45\x46\x41ULT\"5\n\nNormRegion\x12\x13\n\x0f\x41\x43ROSS_CHANNELS\x10\x00\x12\x12\n\x0eWITHIN_CHANNEL\x10\x01\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"Z\n\x13MemoryDataParameter\x12\x12\n\nbatch_size\x18\x01 \x01(\r\x12\x10\n\x08\x63hannels\x18\x02 \x01(\r\x12\x0e\n\x06height\x18\x03 \x01(\r\x12\r\n\x05width\x18\x04 \x01(\r\"\xe8\x08\n\x15MultiBoxLossParameter\x12J\n\rloc_loss_type\x18\x01 \x01(\x0e\x32(.caffe.MultiBoxLossParameter.LocLossType:\tSMOOTH_L1\x12J\n\x0e\x63onf_loss_type\x18\x02 \x01(\x0e\x32).caffe.MultiBoxLossParameter.ConfLossType:\x07SOFTMAX\x12\x15\n\nloc_weight\x18\x03 \x01(\x02:\x01\x31\x12\x13\n\x0bnum_classes\x18\x04 \x01(\r\x12\x1c\n\x0eshare_location\x18\x05 \x01(\x08:\x04true\x12J\n\nmatch_type\x18\x06 \x01(\x0e\x32&.caffe.MultiBoxLossParameter.MatchType:\x0ePER_PREDICTION\x12\x1e\n\x11overlap_threshold\x18\x07 \x01(\x02:\x03\x30.5\x12$\n\x16use_prior_for_matching\x18\x08 \x01(\x08:\x04true\x12\x1e\n\x13\x62\x61\x63kground_label_id\x18\t \x01(\r:\x01\x30\x12\x1e\n\x10use_difficult_gt\x18\n \x01(\x08:\x04true\x12\x15\n\rdo_neg_mining\x18\x0b \x01(\x08\x12\x18\n\rneg_pos_ratio\x18\x0c \x01(\x02:\x01\x33\x12\x18\n\x0bneg_overlap\x18\r \x01(\x02:\x03\x30.5\x12<\n\tcode_type\x18\x0e \x01(\x0e\x32!.caffe.PriorBoxParameter.CodeType:\x06\x43ORNER\x12(\n\x19\x65ncode_variance_in_target\x18\x10 \x01(\x08:\x05\x66\x61lse\x12%\n\x16map_object_to_agnostic\x18\x11 \x01(\x08:\x05\x66\x61lse\x12)\n\x1aignore_cross_boundary_bbox\x18\x12 \x01(\x08:\x05\x66\x61lse\x12\x18\n\tbp_inside\x18\x13 \x01(\x08:\x05\x66\x61lse\x12J\n\x0bmining_type\x18\x14 \x01(\x0e\x32\'.caffe.MultiBoxLossParameter.MiningType:\x0cMAX_NEGATIVE\x12\x38\n\tnms_param\x18\x15 \x01(\x0b\x32%.caffe.NonMaximumSuppressionParameter\x12\x17\n\x0bsample_size\x18\x16 \x01(\x05:\x02\x36\x34\x12 \n\x11use_prior_for_nms\x18\x17 \x01(\x08:\x05\x66\x61lse\"$\n\x0bLocLossType\x12\x06\n\x02L2\x10\x00\x12\r\n\tSMOOTH_L1\x10\x01\")\n\x0c\x43onfLossType\x12\x0b\n\x07SOFTMAX\x10\x00\x12\x0c\n\x08LOGISTIC\x10\x01\".\n\tMatchType\x12\r\n\tBIPARTITE\x10\x00\x12\x12\n\x0ePER_PREDICTION\x10\x01\":\n\nMiningType\x12\x08\n\x04NONE\x10\x00\x12\x10\n\x0cMAX_NEGATIVE\x10\x01\x12\x10\n\x0cHARD_EXAMPLE\x10\x02\"!\n\x10PermuteParameter\x12\r\n\x05order\x18\x01 \x03(\r\"d\n\x0cMVNParameter\x12 \n\x12normalize_variance\x18\x01 \x01(\x08:\x04true\x12\x1e\n\x0f\x61\x63ross_channels\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x12\n\x03\x65ps\x18\x03 \x01(\x02:\x05\x31\x65-09\"5\n\x12ParameterParameter\x12\x1f\n\x05shape\x18\x01 \x01(\x0b\x32\x10.caffe.BlobShape\"\xbb\x03\n\x10PoolingParameter\x12\x35\n\x04pool\x18\x01 \x01(\x0e\x32\".caffe.PoolingParameter.PoolMethod:\x03MAX\x12\x0e\n\x03pad\x18\x04 \x01(\r:\x01\x30\x12\x10\n\x05pad_h\x18\t \x01(\r:\x01\x30\x12\x10\n\x05pad_w\x18\n \x01(\r:\x01\x30\x12\x13\n\x0bkernel_size\x18\x02 \x01(\r\x12\x10\n\x08kernel_h\x18\x05 \x01(\r\x12\x10\n\x08kernel_w\x18\x06 \x01(\r\x12\x11\n\x06stride\x18\x03 \x01(\r:\x01\x31\x12\x10\n\x08stride_h\x18\x07 \x01(\r\x12\x10\n\x08stride_w\x18\x08 \x01(\r\x12\x37\n\x06\x65ngine\x18\x0b \x01(\x0e\x32\x1e.caffe.PoolingParameter.Engine:\x07\x44\x45\x46\x41ULT\x12\x1d\n\x0eglobal_pooling\x18\x0c \x01(\x08:\x05\x66\x61lse\x12\x17\n\tceil_mode\x18\r \x01(\x08:\x04true\".\n\nPoolMethod\x12\x07\n\x03MAX\x10\x00\x12\x07\n\x03\x41VE\x10\x01\x12\x0e\n\nSTOCHASTIC\x10\x02\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"F\n\x0ePowerParameter\x12\x10\n\x05power\x18\x01 \x01(\x02:\x01\x31\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x31\x12\x10\n\x05shift\x18\x03 \x01(\x02:\x01\x30\"\xb5\x02\n\x11PriorBoxParameter\x12\x10\n\x08min_size\x18\x01 \x03(\x02\x12\x10\n\x08max_size\x18\x02 \x03(\x02\x12\x14\n\x0c\x61spect_ratio\x18\x03 \x03(\x02\x12\x12\n\x04\x66lip\x18\x04 \x01(\x08:\x04true\x12\x13\n\x04\x63lip\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x10\n\x08variance\x18\x06 \x03(\x02\x12\x10\n\x08img_size\x18\x07 \x01(\r\x12\r\n\x05img_h\x18\x08 \x01(\r\x12\r\n\x05img_w\x18\t \x01(\r\x12\x0c\n\x04step\x18\n \x01(\x02\x12\x0e\n\x06step_h\x18\x0b \x01(\x02\x12\x0e\n\x06step_w\x18\x0c \x01(\x02\x12\x13\n\x06offset\x18\r \x01(\x02:\x03\x30.5\"8\n\x08\x43odeType\x12\n\n\x06\x43ORNER\x10\x01\x12\x0f\n\x0b\x43\x45NTER_SIZE\x10\x02\x12\x0f\n\x0b\x43ORNER_SIZE\x10\x03\"g\n\x0fPythonParameter\x12\x0e\n\x06module\x18\x01 \x01(\t\x12\r\n\x05layer\x18\x02 \x01(\t\x12\x13\n\tparam_str\x18\x03 \x01(\t:\x00\x12 \n\x11share_in_parallel\x18\x04 \x01(\x08:\x05\x66\x61lse\"\xc0\x01\n\x12RecurrentParameter\x12\x15\n\nnum_output\x18\x01 \x01(\r:\x01\x30\x12-\n\rweight_filler\x18\x02 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x03 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x19\n\ndebug_info\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1c\n\rexpose_hidden\x18\x05 \x01(\x08:\x05\x66\x61lse\"\xad\x01\n\x12ReductionParameter\x12=\n\toperation\x18\x01 \x01(\x0e\x32%.caffe.ReductionParameter.ReductionOp:\x03SUM\x12\x0f\n\x04\x61xis\x18\x02 \x01(\x05:\x01\x30\x12\x10\n\x05\x63oeff\x18\x03 \x01(\x02:\x01\x31\"5\n\x0bReductionOp\x12\x07\n\x03SUM\x10\x01\x12\x08\n\x04\x41SUM\x10\x02\x12\t\n\x05SUMSQ\x10\x03\x12\x08\n\x04MEAN\x10\x04\"\x8d\x01\n\rReLUParameter\x12\x19\n\x0enegative_slope\x18\x01 \x01(\x02:\x01\x30\x12\x34\n\x06\x65ngine\x18\x02 \x01(\x0e\x32\x1b.caffe.ReLUParameter.Engine:\x07\x44\x45\x46\x41ULT\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"Z\n\x10ReshapeParameter\x12\x1f\n\x05shape\x18\x01 \x01(\x0b\x32\x10.caffe.BlobShape\x12\x0f\n\x04\x61xis\x18\x02 \x01(\x05:\x01\x30\x12\x14\n\x08num_axes\x18\x03 \x01(\x05:\x02-1\"Y\n\x13ROIPoolingParameter\x12\x13\n\x08pooled_h\x18\x01 \x01(\r:\x01\x30\x12\x13\n\x08pooled_w\x18\x02 \x01(\r:\x01\x30\x12\x18\n\rspatial_scale\x18\x03 \x01(\x02:\x01\x31\"\xcb\x01\n\x0eScaleParameter\x12\x0f\n\x04\x61xis\x18\x01 \x01(\x05:\x01\x31\x12\x13\n\x08num_axes\x18\x02 \x01(\x05:\x01\x31\x12&\n\x06\x66iller\x18\x03 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x18\n\tbias_term\x18\x04 \x01(\x08:\x05\x66\x61lse\x12+\n\x0b\x62ias_filler\x18\x05 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x11\n\tmin_value\x18\x06 \x01(\x02\x12\x11\n\tmax_value\x18\x07 \x01(\x02\"x\n\x10SigmoidParameter\x12\x37\n\x06\x65ngine\x18\x01 \x01(\x0e\x32\x1e.caffe.SigmoidParameter.Engine:\x07\x44\x45\x46\x41ULT\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\")\n\x15SmoothL1LossParameter\x12\x10\n\x05sigma\x18\x01 \x01(\x02:\x01\x31\"L\n\x0eSliceParameter\x12\x0f\n\x04\x61xis\x18\x03 \x01(\x05:\x01\x31\x12\x13\n\x0bslice_point\x18\x02 \x03(\r\x12\x14\n\tslice_dim\x18\x01 \x01(\r:\x01\x31\"\x89\x01\n\x10SoftmaxParameter\x12\x37\n\x06\x65ngine\x18\x01 \x01(\x0e\x32\x1e.caffe.SoftmaxParameter.Engine:\x07\x44\x45\x46\x41ULT\x12\x0f\n\x04\x61xis\x18\x02 \x01(\x05:\x01\x31\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"r\n\rTanHParameter\x12\x34\n\x06\x65ngine\x18\x01 \x01(\x0e\x32\x1b.caffe.TanHParameter.Engine:\x07\x44\x45\x46\x41ULT\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"/\n\rTileParameter\x12\x0f\n\x04\x61xis\x18\x01 \x01(\x05:\x01\x31\x12\r\n\x05tiles\x18\x02 \x01(\x05\"*\n\x12ThresholdParameter\x12\x14\n\tthreshold\x18\x01 \x01(\x02:\x01\x30\"\xc1\x02\n\x13WindowDataParameter\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x10\n\x05scale\x18\x02 \x01(\x02:\x01\x31\x12\x11\n\tmean_file\x18\x03 \x01(\t\x12\x12\n\nbatch_size\x18\x04 \x01(\r\x12\x14\n\tcrop_size\x18\x05 \x01(\r:\x01\x30\x12\x15\n\x06mirror\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x19\n\x0c\x66g_threshold\x18\x07 \x01(\x02:\x03\x30.5\x12\x19\n\x0c\x62g_threshold\x18\x08 \x01(\x02:\x03\x30.5\x12\x19\n\x0b\x66g_fraction\x18\t \x01(\x02:\x04\x30.25\x12\x16\n\x0b\x63ontext_pad\x18\n \x01(\r:\x01\x30\x12\x17\n\tcrop_mode\x18\x0b \x01(\t:\x04warp\x12\x1b\n\x0c\x63\x61\x63he_images\x18\x0c \x01(\x08:\x05\x66\x61lse\x12\x15\n\x0broot_folder\x18\r \x01(\t:\x00\"\xeb\x01\n\x0cSPPParameter\x12\x16\n\x0epyramid_height\x18\x01 \x01(\r\x12\x31\n\x04pool\x18\x02 \x01(\x0e\x32\x1e.caffe.SPPParameter.PoolMethod:\x03MAX\x12\x33\n\x06\x65ngine\x18\x06 \x01(\x0e\x32\x1a.caffe.SPPParameter.Engine:\x07\x44\x45\x46\x41ULT\".\n\nPoolMethod\x12\x07\n\x03MAX\x10\x00\x12\x07\n\x03\x41VE\x10\x01\x12\x0e\n\nSTOCHASTIC\x10\x02\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"\xdc\x14\n\x10V1LayerParameter\x12\x0e\n\x06\x62ottom\x18\x02 \x03(\t\x12\x0b\n\x03top\x18\x03 \x03(\t\x12\x0c\n\x04name\x18\x04 \x01(\t\x12$\n\x07include\x18  \x03(\x0b\x32\x13.caffe.NetStateRule\x12$\n\x07\x65xclude\x18! \x03(\x0b\x32\x13.caffe.NetStateRule\x12/\n\x04type\x18\x05 \x01(\x0e\x32!.caffe.V1LayerParameter.LayerType\x12\x1f\n\x05\x62lobs\x18\x06 \x03(\x0b\x32\x10.caffe.BlobProto\x12\x0e\n\x05param\x18\xe9\x07 \x03(\t\x12>\n\x0f\x62lob_share_mode\x18\xea\x07 \x03(\x0e\x32$.caffe.V1LayerParameter.DimCheckMode\x12\x10\n\x08\x62lobs_lr\x18\x07 \x03(\x02\x12\x14\n\x0cweight_decay\x18\x08 \x03(\x02\x12\x13\n\x0bloss_weight\x18# \x03(\x02\x12\x30\n\x0e\x61\x63\x63uracy_param\x18\x1b \x01(\x0b\x32\x18.caffe.AccuracyParameter\x12,\n\x0c\x61rgmax_param\x18\x17 \x01(\x0b\x32\x16.caffe.ArgMaxParameter\x12,\n\x0c\x63oncat_param\x18\t \x01(\x0b\x32\x16.caffe.ConcatParameter\x12?\n\x16\x63ontrastive_loss_param\x18( \x01(\x0b\x32\x1f.caffe.ContrastiveLossParameter\x12\x36\n\x11\x63onvolution_param\x18\n \x01(\x0b\x32\x1b.caffe.ConvolutionParameter\x12(\n\ndata_param\x18\x0b \x01(\x0b\x32\x14.caffe.DataParameter\x12.\n\rdropout_param\x18\x0c \x01(\x0b\x32\x17.caffe.DropoutParameter\x12\x33\n\x10\x64ummy_data_param\x18\x1a \x01(\x0b\x32\x19.caffe.DummyDataParameter\x12.\n\reltwise_param\x18\x18 \x01(\x0b\x32\x17.caffe.EltwiseParameter\x12&\n\texp_param\x18) \x01(\x0b\x32\x13.caffe.ExpParameter\x12\x31\n\x0fhdf5_data_param\x18\r \x01(\x0b\x32\x18.caffe.HDF5DataParameter\x12\x35\n\x11hdf5_output_param\x18\x0e \x01(\x0b\x32\x1a.caffe.HDF5OutputParameter\x12\x33\n\x10hinge_loss_param\x18\x1d \x01(\x0b\x32\x19.caffe.HingeLossParameter\x12\x33\n\x10image_data_param\x18\x0f \x01(\x0b\x32\x19.caffe.ImageDataParameter\x12\x39\n\x13infogain_loss_param\x18\x10 \x01(\x0b\x32\x1c.caffe.InfogainLossParameter\x12\x39\n\x13inner_product_param\x18\x11 \x01(\x0b\x32\x1c.caffe.InnerProductParameter\x12&\n\tlrn_param\x18\x12 \x01(\x0b\x32\x13.caffe.LRNParameter\x12\x35\n\x11memory_data_param\x18\x16 \x01(\x0b\x32\x1a.caffe.MemoryDataParameter\x12&\n\tmvn_param\x18\" \x01(\x0b\x32\x13.caffe.MVNParameter\x12.\n\rpooling_param\x18\x13 \x01(\x0b\x32\x17.caffe.PoolingParameter\x12*\n\x0bpower_param\x18\x15 \x01(\x0b\x32\x15.caffe.PowerParameter\x12(\n\nrelu_param\x18\x1e \x01(\x0b\x32\x14.caffe.ReLUParameter\x12.\n\rsigmoid_param\x18& \x01(\x0b\x32\x17.caffe.SigmoidParameter\x12.\n\rsoftmax_param\x18\' \x01(\x0b\x32\x17.caffe.SoftmaxParameter\x12*\n\x0bslice_param\x18\x1f \x01(\x0b\x32\x15.caffe.SliceParameter\x12(\n\ntanh_param\x18% \x01(\x0b\x32\x14.caffe.TanHParameter\x12\x32\n\x0fthreshold_param\x18\x19 \x01(\x0b\x32\x19.caffe.ThresholdParameter\x12\x35\n\x11window_data_param\x18\x14 \x01(\x0b\x32\x1a.caffe.WindowDataParameter\x12\x37\n\x0ftransform_param\x18$ \x01(\x0b\x32\x1e.caffe.TransformationParameter\x12(\n\nloss_param\x18* \x01(\x0b\x32\x14.caffe.LossParameter\x12<\n\x14\x64\x65tection_loss_param\x18\xc8\x01 \x01(\x0b\x32\x1d.caffe.DetectionLossParameter\x12<\n\x14\x65val_detection_param\x18\xc9\x01 \x01(\x0b\x32\x1d.caffe.EvalDetectionParameter\x12&\n\x05layer\x18\x01 \x01(\x0b\x32\x17.caffe.V0LayerParameter\"\xd8\x04\n\tLayerType\x12\x08\n\x04NONE\x10\x00\x12\n\n\x06\x41\x42SVAL\x10#\x12\x0c\n\x08\x41\x43\x43URACY\x10\x01\x12\n\n\x06\x41RGMAX\x10\x1e\x12\x08\n\x04\x42NLL\x10\x02\x12\n\n\x06\x43ONCAT\x10\x03\x12\x14\n\x10\x43ONTRASTIVE_LOSS\x10%\x12\x0f\n\x0b\x43ONVOLUTION\x10\x04\x12\x08\n\x04\x44\x41TA\x10\x05\x12\x11\n\rDECONVOLUTION\x10\'\x12\x0b\n\x07\x44ROPOUT\x10\x06\x12\x0e\n\nDUMMY_DATA\x10 \x12\x12\n\x0e\x45UCLIDEAN_LOSS\x10\x07\x12\x0b\n\x07\x45LTWISE\x10\x19\x12\x07\n\x03\x45XP\x10&\x12\x0b\n\x07\x46LATTEN\x10\x08\x12\r\n\tHDF5_DATA\x10\t\x12\x0f\n\x0bHDF5_OUTPUT\x10\n\x12\x0e\n\nHINGE_LOSS\x10\x1c\x12\n\n\x06IM2COL\x10\x0b\x12\x0e\n\nIMAGE_DATA\x10\x0c\x12\x11\n\rINFOGAIN_LOSS\x10\r\x12\x11\n\rINNER_PRODUCT\x10\x0e\x12\x07\n\x03LRN\x10\x0f\x12\x0f\n\x0bMEMORY_DATA\x10\x1d\x12\x1d\n\x19MULTINOMIAL_LOGISTIC_LOSS\x10\x10\x12\x07\n\x03MVN\x10\"\x12\x0b\n\x07POOLING\x10\x11\x12\t\n\x05POWER\x10\x1a\x12\x08\n\x04RELU\x10\x12\x12\x0b\n\x07SIGMOID\x10\x13\x12\x1e\n\x1aSIGMOID_CROSS_ENTROPY_LOSS\x10\x1b\x12\x0b\n\x07SILENCE\x10$\x12\x0b\n\x07SOFTMAX\x10\x14\x12\x10\n\x0cSOFTMAX_LOSS\x10\x15\x12\t\n\x05SPLIT\x10\x16\x12\t\n\x05SLICE\x10!\x12\x08\n\x04TANH\x10\x17\x12\x0f\n\x0bWINDOW_DATA\x10\x18\x12\r\n\tTHRESHOLD\x10\x1f\"*\n\x0c\x44imCheckMode\x12\n\n\x06STRICT\x10\x00\x12\x0e\n\nPERMISSIVE\x10\x01\"\xfd\x07\n\x10V0LayerParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x12\n\nnum_output\x18\x03 \x01(\r\x12\x16\n\x08\x62iasterm\x18\x04 \x01(\x08:\x04true\x12-\n\rweight_filler\x18\x05 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x06 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x0e\n\x03pad\x18\x07 \x01(\r:\x01\x30\x12\x12\n\nkernelsize\x18\x08 \x01(\r\x12\x10\n\x05group\x18\t \x01(\r:\x01\x31\x12\x11\n\x06stride\x18\n \x01(\r:\x01\x31\x12\x35\n\x04pool\x18\x0b \x01(\x0e\x32\".caffe.V0LayerParameter.PoolMethod:\x03MAX\x12\x1a\n\rdropout_ratio\x18\x0c \x01(\x02:\x03\x30.5\x12\x15\n\nlocal_size\x18\r \x01(\r:\x01\x35\x12\x10\n\x05\x61lpha\x18\x0e \x01(\x02:\x01\x31\x12\x12\n\x04\x62\x65ta\x18\x0f \x01(\x02:\x04\x30.75\x12\x0c\n\x01k\x18\x16 \x01(\x02:\x01\x31\x12\x0e\n\x06source\x18\x10 \x01(\t\x12\x10\n\x05scale\x18\x11 \x01(\x02:\x01\x31\x12\x10\n\x08meanfile\x18\x12 \x01(\t\x12\x11\n\tbatchsize\x18\x13 \x01(\r\x12\x13\n\x08\x63ropsize\x18\x14 \x01(\r:\x01\x30\x12\x15\n\x06mirror\x18\x15 \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x05\x62lobs\x18\x32 \x03(\x0b\x32\x10.caffe.BlobProto\x12\x10\n\x08\x62lobs_lr\x18\x33 \x03(\x02\x12\x14\n\x0cweight_decay\x18\x34 \x03(\x02\x12\x14\n\trand_skip\x18\x35 \x01(\r:\x01\x30\x12\x1d\n\x10\x64\x65t_fg_threshold\x18\x36 \x01(\x02:\x03\x30.5\x12\x1d\n\x10\x64\x65t_bg_threshold\x18\x37 \x01(\x02:\x03\x30.5\x12\x1d\n\x0f\x64\x65t_fg_fraction\x18\x38 \x01(\x02:\x04\x30.25\x12\x1a\n\x0f\x64\x65t_context_pad\x18: \x01(\r:\x01\x30\x12\x1b\n\rdet_crop_mode\x18; \x01(\t:\x04warp\x12\x12\n\x07new_num\x18< \x01(\x05:\x01\x30\x12\x17\n\x0cnew_channels\x18= \x01(\x05:\x01\x30\x12\x15\n\nnew_height\x18> \x01(\x05:\x01\x30\x12\x14\n\tnew_width\x18? \x01(\x05:\x01\x30\x12\x1d\n\x0eshuffle_images\x18@ \x01(\x08:\x05\x66\x61lse\x12\x15\n\nconcat_dim\x18\x41 \x01(\r:\x01\x31\x12\x36\n\x11hdf5_output_param\x18\xe9\x07 \x01(\x0b\x32\x1a.caffe.HDF5OutputParameter\".\n\nPoolMethod\x12\x07\n\x03MAX\x10\x00\x12\x07\n\x03\x41VE\x10\x01\x12\x0e\n\nSTOCHASTIC\x10\x02\"W\n\x0ePReLUParameter\x12&\n\x06\x66iller\x18\x01 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x1d\n\x0e\x63hannel_shared\x18\x02 \x01(\x08:\x05\x66\x61lse\"\xa8\x01\n\x0cRPNParameter\x12\x13\n\x0b\x66\x65\x61t_stride\x18\x01 \x01(\r\x12\x10\n\x08\x62\x61sesize\x18\x02 \x01(\r\x12\r\n\x05scale\x18\x03 \x03(\r\x12\r\n\x05ratio\x18\x04 \x03(\x02\x12\x12\n\nboxminsize\x18\x05 \x01(\r\x12\x14\n\x0cper_nms_topn\x18\t \x01(\r\x12\x15\n\rpost_nms_topn\x18\x0b \x01(\r\x12\x12\n\nnms_thresh\x18\x08 \x01(\x02\"\xbb\x01\n\x12VideoDataParameter\x12?\n\nvideo_type\x18\x01 \x01(\x0e\x32#.caffe.VideoDataParameter.VideoType:\x06WEBCAM\x12\x14\n\tdevice_id\x18\x02 \x01(\x05:\x01\x30\x12\x12\n\nvideo_file\x18\x03 \x01(\t\x12\x16\n\x0bskip_frames\x18\x04 \x01(\r:\x01\x30\"\"\n\tVideoType\x12\n\n\x06WEBCAM\x10\x00\x12\t\n\x05VIDEO\x10\x01\"i\n\x13\x43\x65nterLossParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12-\n\rcenter_filler\x18\x02 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x0f\n\x04\x61xis\x18\x03 \x01(\x05:\x01\x31\"\xd9\x02\n\x1bMarginInnerProductParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12\x43\n\x04type\x18\x02 \x01(\x0e\x32-.caffe.MarginInnerProductParameter.MarginType:\x06SINGLE\x12-\n\rweight_filler\x18\x03 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x0f\n\x04\x61xis\x18\x04 \x01(\x05:\x01\x31\x12\x0f\n\x04\x62\x61se\x18\x05 \x01(\x02:\x01\x31\x12\x10\n\x05gamma\x18\x06 \x01(\x02:\x01\x30\x12\x10\n\x05power\x18\x07 \x01(\x02:\x01\x31\x12\x14\n\titeration\x18\x08 \x01(\x05:\x01\x30\x12\x15\n\nlambda_min\x18\t \x01(\x02:\x01\x30\"?\n\nMarginType\x12\n\n\x06SINGLE\x10\x00\x12\n\n\x06\x44OUBLE\x10\x01\x12\n\n\x06TRIPLE\x10\x02\x12\r\n\tQUADRUPLE\x10\x03\"\x8a\x01\n#AdditiveMarginInnerProductParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12-\n\rweight_filler\x18\x02 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x0f\n\x01m\x18\x03 \x01(\x02:\x04\x30.35\x12\x0f\n\x04\x61xis\x18\x04 \x01(\x05:\x01\x31\"\xad\x04\n\x1e\x44\x65\x66ormableConvolutionParameter\x12\x12\n\nnum_output\x18\x01 \x01(\r\x12\x17\n\tbias_term\x18\x02 \x01(\x08:\x04true\x12\x0b\n\x03pad\x18\x03 \x03(\r\x12\x13\n\x0bkernel_size\x18\x04 \x03(\r\x12\x0e\n\x06stride\x18\x06 \x03(\r\x12\x10\n\x08\x64ilation\x18\x12 \x03(\r\x12\x10\n\x05pad_h\x18\t \x01(\r:\x01\x30\x12\x10\n\x05pad_w\x18\n \x01(\r:\x01\x30\x12\x10\n\x08kernel_h\x18\x0b \x01(\r\x12\x10\n\x08kernel_w\x18\x0c \x01(\r\x12\x10\n\x08stride_h\x18\r \x01(\r\x12\x10\n\x08stride_w\x18\x0e \x01(\r\x12\x10\n\x05group\x18\x05 \x01(\r:\x01\x34\x12\x1b\n\x10\x64\x65\x66ormable_group\x18\x19 \x01(\r:\x01\x34\x12-\n\rweight_filler\x18\x07 \x01(\x0b\x32\x16.caffe.FillerParameter\x12+\n\x0b\x62ias_filler\x18\x08 \x01(\x0b\x32\x16.caffe.FillerParameter\x12\x45\n\x06\x65ngine\x18\x0f \x01(\x0e\x32,.caffe.DeformableConvolutionParameter.Engine:\x07\x44\x45\x46\x41ULT\x12\x0f\n\x04\x61xis\x18\x10 \x01(\x05:\x01\x31\x12\x1e\n\x0f\x66orce_nd_im2col\x18\x11 \x01(\x08:\x05\x66\x61lse\"+\n\x06\x45ngine\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\t\n\x05\x43\x41\x46\x46\x45\x10\x01\x12\t\n\x05\x43UDNN\x10\x02\"K\n\x19LabelSpecificAddParameter\x12\x0f\n\x04\x62ias\x18\x01 \x01(\x02:\x01\x30\x12\x1d\n\x0etransform_test\x18\x02 \x01(\x08:\x05\x66\x61lse\"\xed\x01\n\x15\x43hannelScaleParameter\x12\x18\n\ndo_forward\x18\x01 \x01(\x08:\x04true\x12!\n\x13\x64o_backward_feature\x18\x02 \x01(\x08:\x04true\x12\x1f\n\x11\x64o_backward_scale\x18\x03 \x01(\x08:\x04true\x12\x1b\n\x0cglobal_scale\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1e\n\x10max_global_scale\x18\x05 \x01(\x02:\x04\x31\x30\x30\x30\x12\x1b\n\x10min_global_scale\x18\x06 \x01(\x02:\x01\x30\x12\x1c\n\x11init_global_scale\x18\x07 \x01(\x02:\x01\x31\"C\n\x12\x43osinAddmParameter\x12\x0e\n\x01m\x18\x01 \x01(\x02:\x03\x30.5\x12\x1d\n\x0etransform_test\x18\x02 \x01(\x08:\x05\x66\x61lse\"A\n\x12\x43osinMulmParameter\x12\x0c\n\x01m\x18\x01 \x01(\x02:\x01\x34\x12\x1d\n\x0etransform_test\x18\x02 \x01(\x08:\x05\x66\x61lse\"r\n\x1b\x43oupledClusterLossParameter\x12\x11\n\x06margin\x18\x01 \x01(\x02:\x01\x31\x12\x15\n\ngroup_size\x18\x02 \x01(\x05:\x01\x33\x12\x10\n\x05scale\x18\x03 \x01(\x02:\x01\x31\x12\x17\n\x08log_flag\x18\x04 \x01(\x08:\x05\x66\x61lse\"R\n\x14TripletLossParameter\x12\x11\n\x06margin\x18\x01 \x01(\x02:\x01\x31\x12\x15\n\ngroup_size\x18\x02 \x01(\x05:\x01\x33\x12\x10\n\x05scale\x18\x03 \x01(\x02:\x01\x31\"\xe2\x01\n\x17GeneralTripletParameter\x12\x13\n\x06margin\x18\x01 \x01(\x02:\x03\x30.2\x12\x1d\n\x0f\x61\x64\x64_center_loss\x18\x02 \x01(\x08:\x04true\x12\x1b\n\x0chardest_only\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x0epositive_first\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x14positive_upper_bound\x18\x05 \x01(\x02:\x01\x31\x12\x1a\n\x0fpositive_weight\x18\x06 \x01(\x02:\x01\x31\x12\x1a\n\x0fnegative_weight\x18\x07 \x01(\x02:\x01\x31\"W\n\x11ROIAlignParameter\x12\x13\n\x08pooled_h\x18\x01 \x01(\r:\x01\x30\x12\x13\n\x08pooled_w\x18\x02 \x01(\r:\x01\x30\x12\x18\n\rspatial_scale\x18\x03 \x01(\x02:\x01\x31*\x1c\n\x05Phase\x12\t\n\x05TRAIN\x10\x00\x12\x08\n\x04TEST\x10\x01')
)

_PHASE = _descriptor.EnumDescriptor(
  name='Phase',
  full_name='caffe.Phase',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TRAIN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEST', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=29109,
  serialized_end=29137,
)
_sym_db.RegisterEnumDescriptor(_PHASE)

Phase = enum_type_wrapper.EnumTypeWrapper(_PHASE)
TRAIN = 0
TEST = 1


_EMITCONSTRAINT_EMITTYPE = _descriptor.EnumDescriptor(
  name='EmitType',
  full_name='caffe.EmitConstraint.EmitType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CENTER', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MIN_OVERLAP', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1162,
  serialized_end=1201,
)
_sym_db.RegisterEnumDescriptor(_EMITCONSTRAINT_EMITTYPE)

_ANNOTATEDDATUM_ANNOTATIONTYPE = _descriptor.EnumDescriptor(
  name='AnnotationType',
  full_name='caffe.AnnotatedDatum.AnnotationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BBOX', index=0, number=0,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1645,
  serialized_end=1671,
)
_sym_db.RegisterEnumDescriptor(_ANNOTATEDDATUM_ANNOTATIONTYPE)

_FILLERPARAMETER_VARIANCENORM = _descriptor.EnumDescriptor(
  name='VarianceNorm',
  full_name='caffe.FillerParameter.VarianceNorm',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FAN_IN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FAN_OUT', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AVERAGE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2058,
  serialized_end=2110,
)
_sym_db.RegisterEnumDescriptor(_FILLERPARAMETER_VARIANCENORM)

_SOLVERPARAMETER_SNAPSHOTFORMAT = _descriptor.EnumDescriptor(
  name='SnapshotFormat',
  full_name='caffe.SolverParameter.SnapshotFormat',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='HDF5', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BINARYPROTO', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=3568,
  serialized_end=3611,
)
_sym_db.RegisterEnumDescriptor(_SOLVERPARAMETER_SNAPSHOTFORMAT)

_SOLVERPARAMETER_SOLVERMODE = _descriptor.EnumDescriptor(
  name='SolverMode',
  full_name='caffe.SolverParameter.SolverMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CPU', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GPU', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=3613,
  serialized_end=3643,
)
_sym_db.RegisterEnumDescriptor(_SOLVERPARAMETER_SOLVERMODE)

_SOLVERPARAMETER_SOLVERTYPE = _descriptor.EnumDescriptor(
  name='SolverType',
  full_name='caffe.SolverParameter.SolverType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SGD', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NESTEROV', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ADAGRAD', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RMSPROP', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ADADELTA', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ADAM', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=3645,
  serialized_end=3730,
)
_sym_db.RegisterEnumDescriptor(_SOLVERPARAMETER_SOLVERTYPE)

_PARAMSPEC_DIMCHECKMODE = _descriptor.EnumDescriptor(
  name='DimCheckMode',
  full_name='caffe.ParamSpec.DimCheckMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STRICT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PERMISSIVE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=4491,
  serialized_end=4533,
)
_sym_db.RegisterEnumDescriptor(_PARAMSPEC_DIMCHECKMODE)

_BNPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.BNParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_BNPARAMETER_ENGINE)

_FOCALLOSSPARAMETER_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='caffe.FocalLossParameter.Type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ORIGIN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LINEAR', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=11083,
  serialized_end=11113,
)
_sym_db.RegisterEnumDescriptor(_FOCALLOSSPARAMETER_TYPE)

_RESIZEPARAMETER_RESIZE_MODE = _descriptor.EnumDescriptor(
  name='Resize_mode',
  full_name='caffe.ResizeParameter.Resize_mode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='WARP', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIT_SMALL_SIZE', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIT_LARGE_SIZE_AND_PAD', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=11899,
  serialized_end=11970,
)
_sym_db.RegisterEnumDescriptor(_RESIZEPARAMETER_RESIZE_MODE)

_RESIZEPARAMETER_PAD_MODE = _descriptor.EnumDescriptor(
  name='Pad_mode',
  full_name='caffe.ResizeParameter.Pad_mode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CONSTANT', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MIRRORED', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REPEAT_NEAREST', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=11972,
  serialized_end=12030,
)
_sym_db.RegisterEnumDescriptor(_RESIZEPARAMETER_PAD_MODE)

_RESIZEPARAMETER_INTERP_MODE = _descriptor.EnumDescriptor(
  name='Interp_mode',
  full_name='caffe.ResizeParameter.Interp_mode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LINEAR', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AREA', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NEAREST', index=2, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUBIC', index=3, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANCZOS4', index=4, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=12032,
  serialized_end=12105,
)
_sym_db.RegisterEnumDescriptor(_RESIZEPARAMETER_INTERP_MODE)

_LOSSPARAMETER_NORMALIZATIONMODE = _descriptor.EnumDescriptor(
  name='NormalizationMode',
  full_name='caffe.LossParameter.NormalizationMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FULL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VALID', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BATCH_SIZE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NONE', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=13052,
  serialized_end=13118,
)
_sym_db.RegisterEnumDescriptor(_LOSSPARAMETER_NORMALIZATIONMODE)

_EVALDETECTIONPARAMETER_SCORETYPE = _descriptor.EnumDescriptor(
  name='ScoreType',
  full_name='caffe.EvalDetectionParameter.ScoreType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OBJ', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROB', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTIPLY', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=14582,
  serialized_end=14626,
)
_sym_db.RegisterEnumDescriptor(_EVALDETECTIONPARAMETER_SCORETYPE)

_CONVOLUTIONPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.ConvolutionParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_CONVOLUTIONPARAMETER_ENGINE)

_DATAPARAMETER_DB = _descriptor.EnumDescriptor(
  name='DB',
  full_name='caffe.DataParameter.DB',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LEVELDB', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LMDB', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=15469,
  serialized_end=15496,
)
_sym_db.RegisterEnumDescriptor(_DATAPARAMETER_DB)

_ELTWISEPARAMETER_ELTWISEOP = _descriptor.EnumDescriptor(
  name='EltwiseOp',
  full_name='caffe.EltwiseParameter.EltwiseOp',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PROD', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUM', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MAX', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=16856,
  serialized_end=16895,
)
_sym_db.RegisterEnumDescriptor(_ELTWISEPARAMETER_ELTWISEOP)

_HINGELOSSPARAMETER_NORM = _descriptor.EnumDescriptor(
  name='Norm',
  full_name='caffe.HingeLossParameter.Norm',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='L1', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='L2', index=1, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=17430,
  serialized_end=17452,
)
_sym_db.RegisterEnumDescriptor(_HINGELOSSPARAMETER_NORM)

_LRNPARAMETER_NORMREGION = _descriptor.EnumDescriptor(
  name='NormRegion',
  full_name='caffe.LRNParameter.NormRegion',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ACROSS_CHANNELS', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WITHIN_CHANNEL', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=18345,
  serialized_end=18398,
)
_sym_db.RegisterEnumDescriptor(_LRNPARAMETER_NORMREGION)

_LRNPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.LRNParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_LRNPARAMETER_ENGINE)

_MULTIBOXLOSSPARAMETER_LOCLOSSTYPE = _descriptor.EnumDescriptor(
  name='LocLossType',
  full_name='caffe.MultiBoxLossParameter.LocLossType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='L2', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SMOOTH_L1', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=19479,
  serialized_end=19515,
)
_sym_db.RegisterEnumDescriptor(_MULTIBOXLOSSPARAMETER_LOCLOSSTYPE)

_MULTIBOXLOSSPARAMETER_CONFLOSSTYPE = _descriptor.EnumDescriptor(
  name='ConfLossType',
  full_name='caffe.MultiBoxLossParameter.ConfLossType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SOFTMAX', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOGISTIC', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=19517,
  serialized_end=19558,
)
_sym_db.RegisterEnumDescriptor(_MULTIBOXLOSSPARAMETER_CONFLOSSTYPE)

_MULTIBOXLOSSPARAMETER_MATCHTYPE = _descriptor.EnumDescriptor(
  name='MatchType',
  full_name='caffe.MultiBoxLossParameter.MatchType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BIPARTITE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PER_PREDICTION', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=19560,
  serialized_end=19606,
)
_sym_db.RegisterEnumDescriptor(_MULTIBOXLOSSPARAMETER_MATCHTYPE)

_MULTIBOXLOSSPARAMETER_MININGTYPE = _descriptor.EnumDescriptor(
  name='MiningType',
  full_name='caffe.MultiBoxLossParameter.MiningType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MAX_NEGATIVE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HARD_EXAMPLE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=19608,
  serialized_end=19666,
)
_sym_db.RegisterEnumDescriptor(_MULTIBOXLOSSPARAMETER_MININGTYPE)

_POOLINGPARAMETER_POOLMETHOD = _descriptor.EnumDescriptor(
  name='PoolMethod',
  full_name='caffe.PoolingParameter.PoolMethod',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MAX', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AVE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STOCHASTIC', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=20213,
  serialized_end=20259,
)
_sym_db.RegisterEnumDescriptor(_POOLINGPARAMETER_POOLMETHOD)

_POOLINGPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.PoolingParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_POOLINGPARAMETER_ENGINE)

_PRIORBOXPARAMETER_CODETYPE = _descriptor.EnumDescriptor(
  name='CodeType',
  full_name='caffe.PriorBoxParameter.CodeType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CORNER', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CENTER_SIZE', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CORNER_SIZE', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=20632,
  serialized_end=20688,
)
_sym_db.RegisterEnumDescriptor(_PRIORBOXPARAMETER_CODETYPE)

_REDUCTIONPARAMETER_REDUCTIONOP = _descriptor.EnumDescriptor(
  name='ReductionOp',
  full_name='caffe.ReductionParameter.ReductionOp',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SUM', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ASUM', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUMSQ', index=2, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MEAN', index=3, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=21111,
  serialized_end=21164,
)
_sym_db.RegisterEnumDescriptor(_REDUCTIONPARAMETER_REDUCTIONOP)

_RELUPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.ReLUParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_RELUPARAMETER_ENGINE)

_SIGMOIDPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.SigmoidParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_SIGMOIDPARAMETER_ENGINE)

_SOFTMAXPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.SoftmaxParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_SOFTMAXPARAMETER_ENGINE)

_TANHPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.TanHParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_TANHPARAMETER_ENGINE)

_SPPPARAMETER_POOLMETHOD = _descriptor.EnumDescriptor(
  name='PoolMethod',
  full_name='caffe.SPPParameter.PoolMethod',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MAX', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AVE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STOCHASTIC', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=20213,
  serialized_end=20259,
)
_sym_db.RegisterEnumDescriptor(_SPPPARAMETER_POOLMETHOD)

_SPPPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.SPPParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_SPPPARAMETER_ENGINE)

_V1LAYERPARAMETER_LAYERTYPE = _descriptor.EnumDescriptor(
  name='LayerType',
  full_name='caffe.V1LayerParameter.LayerType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ABSVAL', index=1, number=35,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACCURACY', index=2, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ARGMAX', index=3, number=30,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BNLL', index=4, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONCAT', index=5, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONTRASTIVE_LOSS', index=6, number=37,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONVOLUTION', index=7, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DATA', index=8, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DECONVOLUTION', index=9, number=39,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DROPOUT', index=10, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DUMMY_DATA', index=11, number=32,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EUCLIDEAN_LOSS', index=12, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ELTWISE', index=13, number=25,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EXP', index=14, number=38,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLATTEN', index=15, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDF5_DATA', index=16, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDF5_OUTPUT', index=17, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HINGE_LOSS', index=18, number=28,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IM2COL', index=19, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMAGE_DATA', index=20, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INFOGAIN_LOSS', index=21, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INNER_PRODUCT', index=22, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LRN', index=23, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MEMORY_DATA', index=24, number=29,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTINOMIAL_LOGISTIC_LOSS', index=25, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MVN', index=26, number=34,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='POOLING', index=27, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='POWER', index=28, number=26,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RELU', index=29, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=30, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID_CROSS_ENTROPY_LOSS', index=31, number=27,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SILENCE', index=32, number=36,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SOFTMAX', index=33, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SOFTMAX_LOSS', index=34, number=21,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SPLIT', index=35, number=22,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SLICE', index=36, number=33,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TANH', index=37, number=23,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WINDOW_DATA', index=38, number=24,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='THRESHOLD', index=39, number=31,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=24862,
  serialized_end=25462,
)
_sym_db.RegisterEnumDescriptor(_V1LAYERPARAMETER_LAYERTYPE)

_V1LAYERPARAMETER_DIMCHECKMODE = _descriptor.EnumDescriptor(
  name='DimCheckMode',
  full_name='caffe.V1LayerParameter.DimCheckMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STRICT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PERMISSIVE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=4491,
  serialized_end=4533,
)
_sym_db.RegisterEnumDescriptor(_V1LAYERPARAMETER_DIMCHECKMODE)

_V0LAYERPARAMETER_POOLMETHOD = _descriptor.EnumDescriptor(
  name='PoolMethod',
  full_name='caffe.V0LayerParameter.PoolMethod',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MAX', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AVE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STOCHASTIC', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=20213,
  serialized_end=20259,
)
_sym_db.RegisterEnumDescriptor(_V0LAYERPARAMETER_POOLMETHOD)

_VIDEODATAPARAMETER_VIDEOTYPE = _descriptor.EnumDescriptor(
  name='VideoType',
  full_name='caffe.VideoDataParameter.VideoType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='WEBCAM', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VIDEO', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=26946,
  serialized_end=26980,
)
_sym_db.RegisterEnumDescriptor(_VIDEODATAPARAMETER_VIDEOTYPE)

_MARGININNERPRODUCTPARAMETER_MARGINTYPE = _descriptor.EnumDescriptor(
  name='MarginType',
  full_name='caffe.MarginInnerProductParameter.MarginType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SINGLE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOUBLE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TRIPLE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='QUADRUPLE', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=27372,
  serialized_end=27435,
)
_sym_db.RegisterEnumDescriptor(_MARGININNERPRODUCTPARAMETER_MARGINTYPE)

_DEFORMABLECONVOLUTIONPARAMETER_ENGINE = _descriptor.EnumDescriptor(
  name='Engine',
  full_name='caffe.DeformableConvolutionParameter.Engine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAFFE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CUDNN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=10905,
  serialized_end=10948,
)
_sym_db.RegisterEnumDescriptor(_DEFORMABLECONVOLUTIONPARAMETER_ENGINE)


_BLOBSHAPE = _descriptor.Descriptor(
  name='BlobShape',
  full_name='caffe.BlobShape',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dim', full_name='caffe.BlobShape.dim', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=50,
)


_BLOBPROTO = _descriptor.Descriptor(
  name='BlobProto',
  full_name='caffe.BlobProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='caffe.BlobProto.shape', index=0,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='caffe.BlobProto.data', index=1,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='diff', full_name='caffe.BlobProto.diff', index=2,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='double_data', full_name='caffe.BlobProto.double_data', index=3,
      number=8, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='double_diff', full_name='caffe.BlobProto.double_diff', index=4,
      number=9, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num', full_name='caffe.BlobProto.num', index=5,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channels', full_name='caffe.BlobProto.channels', index=6,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='caffe.BlobProto.height', index=7,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffe.BlobProto.width', index=8,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=53,
  serialized_end=257,
)


_BLOBPROTOVECTOR = _descriptor.Descriptor(
  name='BlobProtoVector',
  full_name='caffe.BlobProtoVector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='blobs', full_name='caffe.BlobProtoVector.blobs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=259,
  serialized_end=309,
)


_DATUM = _descriptor.Descriptor(
  name='Datum',
  full_name='caffe.Datum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='channels', full_name='caffe.Datum.channels', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='caffe.Datum.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffe.Datum.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='caffe.Datum.data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='caffe.Datum.label', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_data', full_name='caffe.Datum.float_data', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='encoded', full_name='caffe.Datum.encoded', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='labels', full_name='caffe.Datum.labels', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=312,
  serialized_end=457,
)


_LABELMAPITEM = _descriptor.Descriptor(
  name='LabelMapItem',
  full_name='caffe.LabelMapItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe.LabelMapItem.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='caffe.LabelMapItem.label', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='caffe.LabelMapItem.display_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=459,
  serialized_end=524,
)


_LABELMAP = _descriptor.Descriptor(
  name='LabelMap',
  full_name='caffe.LabelMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='item', full_name='caffe.LabelMap.item', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=526,
  serialized_end=571,
)


_SAMPLER = _descriptor.Descriptor(
  name='Sampler',
  full_name='caffe.Sampler',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_scale', full_name='caffe.Sampler.min_scale', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_scale', full_name='caffe.Sampler.max_scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_aspect_ratio', full_name='caffe.Sampler.min_aspect_ratio', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_aspect_ratio', full_name='caffe.Sampler.max_aspect_ratio', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=573,
  serialized_end=684,
)


_SAMPLECONSTRAINT = _descriptor.Descriptor(
  name='SampleConstraint',
  full_name='caffe.SampleConstraint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_jaccard_overlap', full_name='caffe.SampleConstraint.min_jaccard_overlap', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_jaccard_overlap', full_name='caffe.SampleConstraint.max_jaccard_overlap', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_sample_coverage', full_name='caffe.SampleConstraint.min_sample_coverage', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_sample_coverage', full_name='caffe.SampleConstraint.max_sample_coverage', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_object_coverage', full_name='caffe.SampleConstraint.min_object_coverage', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_object_coverage', full_name='caffe.SampleConstraint.max_object_coverage', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=687,
  serialized_end=879,
)


_BATCHSAMPLER = _descriptor.Descriptor(
  name='BatchSampler',
  full_name='caffe.BatchSampler',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='use_original_image', full_name='caffe.BatchSampler.use_original_image', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sampler', full_name='caffe.BatchSampler.sampler', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample_constraint', full_name='caffe.BatchSampler.sample_constraint', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_sample', full_name='caffe.BatchSampler.max_sample', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_trials', full_name='caffe.BatchSampler.max_trials', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=882,
  serialized_end=1060,
)


_EMITCONSTRAINT = _descriptor.Descriptor(
  name='EmitConstraint',
  full_name='caffe.EmitConstraint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='emit_type', full_name='caffe.EmitConstraint.emit_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='emit_overlap', full_name='caffe.EmitConstraint.emit_overlap', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _EMITCONSTRAINT_EMITTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1063,
  serialized_end=1201,
)


_NORMALIZEDBBOX = _descriptor.Descriptor(
  name='NormalizedBBox',
  full_name='caffe.NormalizedBBox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='xmin', full_name='caffe.NormalizedBBox.xmin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ymin', full_name='caffe.NormalizedBBox.ymin', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='xmax', full_name='caffe.NormalizedBBox.xmax', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ymax', full_name='caffe.NormalizedBBox.ymax', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='caffe.NormalizedBBox.label', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='difficult', full_name='caffe.NormalizedBBox.difficult', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='caffe.NormalizedBBox.score', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size', full_name='caffe.NormalizedBBox.size', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1204,
  serialized_end=1339,
)


_ANNOTATION = _descriptor.Descriptor(
  name='Annotation',
  full_name='caffe.Annotation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='instance_id', full_name='caffe.Annotation.instance_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bbox', full_name='caffe.Annotation.bbox', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1341,
  serialized_end=1414,
)


_ANNOTATIONGROUP = _descriptor.Descriptor(
  name='AnnotationGroup',
  full_name='caffe.AnnotationGroup',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='group_label', full_name='caffe.AnnotationGroup.group_label', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='annotation', full_name='caffe.AnnotationGroup.annotation', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1416,
  serialized_end=1493,
)


_ANNOTATEDDATUM = _descriptor.Descriptor(
  name='AnnotatedDatum',
  full_name='caffe.AnnotatedDatum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='datum', full_name='caffe.AnnotatedDatum.datum', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.AnnotatedDatum.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='annotation_group', full_name='caffe.AnnotatedDatum.annotation_group', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ANNOTATEDDATUM_ANNOTATIONTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1496,
  serialized_end=1671,
)


_MTCNNBBOX = _descriptor.Descriptor(
  name='MTCNNBBox',
  full_name='caffe.MTCNNBBox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='xmin', full_name='caffe.MTCNNBBox.xmin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ymin', full_name='caffe.MTCNNBBox.ymin', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='xmax', full_name='caffe.MTCNNBBox.xmax', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ymax', full_name='caffe.MTCNNBBox.ymax', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1673,
  serialized_end=1740,
)


_MTCNNDATUM = _descriptor.Descriptor(
  name='MTCNNDatum',
  full_name='caffe.MTCNNDatum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='datum', full_name='caffe.MTCNNDatum.datum', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roi', full_name='caffe.MTCNNDatum.roi', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pts', full_name='caffe.MTCNNDatum.pts', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1742,
  serialized_end=1827,
)


_FILLERPARAMETER = _descriptor.Descriptor(
  name='FillerParameter',
  full_name='caffe.FillerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.FillerParameter.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("constant").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='caffe.FillerParameter.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min', full_name='caffe.FillerParameter.min', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max', full_name='caffe.FillerParameter.max', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean', full_name='caffe.FillerParameter.mean', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std', full_name='caffe.FillerParameter.std', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sparse', full_name='caffe.FillerParameter.sparse', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='variance_norm', full_name='caffe.FillerParameter.variance_norm', index=7,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='file', full_name='caffe.FillerParameter.file', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _FILLERPARAMETER_VARIANCENORM,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1830,
  serialized_end=2110,
)


_NETPARAMETER = _descriptor.Descriptor(
  name='NetParameter',
  full_name='caffe.NetParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe.NetParameter.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input', full_name='caffe.NetParameter.input', index=1,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_shape', full_name='caffe.NetParameter.input_shape', index=2,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_dim', full_name='caffe.NetParameter.input_dim', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_backward', full_name='caffe.NetParameter.force_backward', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='caffe.NetParameter.state', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='debug_info', full_name='caffe.NetParameter.debug_info', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer', full_name='caffe.NetParameter.layer', index=7,
      number=100, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layers', full_name='caffe.NetParameter.layers', index=8,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2113,
  serialized_end=2383,
)


_SOLVERPARAMETER = _descriptor.Descriptor(
  name='SolverParameter',
  full_name='caffe.SolverParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='net', full_name='caffe.SolverParameter.net', index=0,
      number=24, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='net_param', full_name='caffe.SolverParameter.net_param', index=1,
      number=25, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_net', full_name='caffe.SolverParameter.train_net', index=2,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_net', full_name='caffe.SolverParameter.test_net', index=3,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_net_param', full_name='caffe.SolverParameter.train_net_param', index=4,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_net_param', full_name='caffe.SolverParameter.test_net_param', index=5,
      number=22, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_state', full_name='caffe.SolverParameter.train_state', index=6,
      number=26, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_state', full_name='caffe.SolverParameter.test_state', index=7,
      number=27, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_iter', full_name='caffe.SolverParameter.test_iter', index=8,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_interval', full_name='caffe.SolverParameter.test_interval', index=9,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_compute_loss', full_name='caffe.SolverParameter.test_compute_loss', index=10,
      number=19, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_initialization', full_name='caffe.SolverParameter.test_initialization', index=11,
      number=32, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='base_lr', full_name='caffe.SolverParameter.base_lr', index=12,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='display', full_name='caffe.SolverParameter.display', index=13,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='average_loss', full_name='caffe.SolverParameter.average_loss', index=14,
      number=33, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_iter', full_name='caffe.SolverParameter.max_iter', index=15,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='iter_size', full_name='caffe.SolverParameter.iter_size', index=16,
      number=36, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lr_policy', full_name='caffe.SolverParameter.lr_policy', index=17,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gamma', full_name='caffe.SolverParameter.gamma', index=18,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power', full_name='caffe.SolverParameter.power', index=19,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='caffe.SolverParameter.momentum', index=20,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='caffe.SolverParameter.weight_decay', index=21,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='regularization_type', full_name='caffe.SolverParameter.regularization_type', index=22,
      number=29, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("L2").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stepsize', full_name='caffe.SolverParameter.stepsize', index=23,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stepvalue', full_name='caffe.SolverParameter.stepvalue', index=24,
      number=34, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stagelr', full_name='caffe.SolverParameter.stagelr', index=25,
      number=50, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stageiter', full_name='caffe.SolverParameter.stageiter', index=26,
      number=51, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clip_gradients', full_name='caffe.SolverParameter.clip_gradients', index=27,
      number=35, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot', full_name='caffe.SolverParameter.snapshot', index=28,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot_prefix', full_name='caffe.SolverParameter.snapshot_prefix', index=29,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot_diff', full_name='caffe.SolverParameter.snapshot_diff', index=30,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot_format', full_name='caffe.SolverParameter.snapshot_format', index=31,
      number=37, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='solver_mode', full_name='caffe.SolverParameter.solver_mode', index=32,
      number=17, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_id', full_name='caffe.SolverParameter.device_id', index=33,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_seed', full_name='caffe.SolverParameter.random_seed', index=34,
      number=20, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.SolverParameter.type', index=35,
      number=40, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("SGD").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delta', full_name='caffe.SolverParameter.delta', index=36,
      number=31, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1e-08),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum2', full_name='caffe.SolverParameter.momentum2', index=37,
      number=39, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.999),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rms_decay', full_name='caffe.SolverParameter.rms_decay', index=38,
      number=38, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='debug_info', full_name='caffe.SolverParameter.debug_info', index=39,
      number=23, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='snapshot_after_train', full_name='caffe.SolverParameter.snapshot_after_train', index=40,
      number=28, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='solver_type', full_name='caffe.SolverParameter.solver_type', index=41,
      number=30, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SOLVERPARAMETER_SNAPSHOTFORMAT,
    _SOLVERPARAMETER_SOLVERMODE,
    _SOLVERPARAMETER_SOLVERTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2386,
  serialized_end=3730,
)


_SOLVERSTATE = _descriptor.Descriptor(
  name='SolverState',
  full_name='caffe.SolverState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='iter', full_name='caffe.SolverState.iter', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learned_net', full_name='caffe.SolverState.learned_net', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='history', full_name='caffe.SolverState.history', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='current_step', full_name='caffe.SolverState.current_step', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3732,
  serialized_end=3840,
)


_NETSTATE = _descriptor.Descriptor(
  name='NetState',
  full_name='caffe.NetState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phase', full_name='caffe.NetState.phase', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='level', full_name='caffe.NetState.level', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stage', full_name='caffe.NetState.stage', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3842,
  serialized_end=3920,
)


_NETSTATERULE = _descriptor.Descriptor(
  name='NetStateRule',
  full_name='caffe.NetStateRule',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phase', full_name='caffe.NetStateRule.phase', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_level', full_name='caffe.NetStateRule.min_level', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_level', full_name='caffe.NetStateRule.max_level', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stage', full_name='caffe.NetStateRule.stage', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='not_stage', full_name='caffe.NetStateRule.not_stage', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=3922,
  serialized_end=4037,
)


_SPATIALTRANSFORMERPARAMETER = _descriptor.Descriptor(
  name='SpatialTransformerParameter',
  full_name='caffe.SpatialTransformerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='transform_type', full_name='caffe.SpatialTransformerParameter.transform_type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("affine").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sampler_type', full_name='caffe.SpatialTransformerParameter.sampler_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("bilinear").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_H', full_name='caffe.SpatialTransformerParameter.output_H', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_W', full_name='caffe.SpatialTransformerParameter.output_W', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='to_compute_dU', full_name='caffe.SpatialTransformerParameter.to_compute_dU', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta_1_1', full_name='caffe.SpatialTransformerParameter.theta_1_1', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta_1_2', full_name='caffe.SpatialTransformerParameter.theta_1_2', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta_1_3', full_name='caffe.SpatialTransformerParameter.theta_1_3', index=7,
      number=8, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta_2_1', full_name='caffe.SpatialTransformerParameter.theta_2_1', index=8,
      number=9, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta_2_2', full_name='caffe.SpatialTransformerParameter.theta_2_2', index=9,
      number=10, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta_2_3', full_name='caffe.SpatialTransformerParameter.theta_2_3', index=10,
      number=11, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4040,
  serialized_end=4312,
)


_STLOSSPARAMETER = _descriptor.Descriptor(
  name='STLossParameter',
  full_name='caffe.STLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_H', full_name='caffe.STLossParameter.output_H', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_W', full_name='caffe.STLossParameter.output_W', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4314,
  serialized_end=4367,
)


_PARAMSPEC = _descriptor.Descriptor(
  name='ParamSpec',
  full_name='caffe.ParamSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe.ParamSpec.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='share_mode', full_name='caffe.ParamSpec.share_mode', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lr_mult', full_name='caffe.ParamSpec.lr_mult', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decay_mult', full_name='caffe.ParamSpec.decay_mult', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _PARAMSPEC_DIMCHECKMODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4370,
  serialized_end=4533,
)


_LAYERPARAMETER = _descriptor.Descriptor(
  name='LayerParameter',
  full_name='caffe.LayerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe.LayerParameter.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.LayerParameter.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bottom', full_name='caffe.LayerParameter.bottom', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='top', full_name='caffe.LayerParameter.top', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='phase', full_name='caffe.LayerParameter.phase', index=4,
      number=10, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_weight', full_name='caffe.LayerParameter.loss_weight', index=5,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='param', full_name='caffe.LayerParameter.param', index=6,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blobs', full_name='caffe.LayerParameter.blobs', index=7,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='propagate_down', full_name='caffe.LayerParameter.propagate_down', index=8,
      number=11, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='include', full_name='caffe.LayerParameter.include', index=9,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='exclude', full_name='caffe.LayerParameter.exclude', index=10,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_param', full_name='caffe.LayerParameter.transform_param', index=11,
      number=100, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_param', full_name='caffe.LayerParameter.loss_param', index=12,
      number=101, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_loss_param', full_name='caffe.LayerParameter.detection_loss_param', index=13,
      number=200, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eval_detection_param', full_name='caffe.LayerParameter.eval_detection_param', index=14,
      number=201, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='region_loss_param', full_name='caffe.LayerParameter.region_loss_param', index=15,
      number=202, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reorg_param', full_name='caffe.LayerParameter.reorg_param', index=16,
      number=203, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='accuracy_param', full_name='caffe.LayerParameter.accuracy_param', index=17,
      number=102, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='argmax_param', full_name='caffe.LayerParameter.argmax_param', index=18,
      number=103, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_norm_param', full_name='caffe.LayerParameter.batch_norm_param', index=19,
      number=139, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_param', full_name='caffe.LayerParameter.bias_param', index=20,
      number=141, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='concat_param', full_name='caffe.LayerParameter.concat_param', index=21,
      number=104, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='contrastive_loss_param', full_name='caffe.LayerParameter.contrastive_loss_param', index=22,
      number=105, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='convolution_param', full_name='caffe.LayerParameter.convolution_param', index=23,
      number=106, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_param', full_name='caffe.LayerParameter.data_param', index=24,
      number=107, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dropout_param', full_name='caffe.LayerParameter.dropout_param', index=25,
      number=108, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dummy_data_param', full_name='caffe.LayerParameter.dummy_data_param', index=26,
      number=109, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eltwise_param', full_name='caffe.LayerParameter.eltwise_param', index=27,
      number=110, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='elu_param', full_name='caffe.LayerParameter.elu_param', index=28,
      number=140, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='embed_param', full_name='caffe.LayerParameter.embed_param', index=29,
      number=137, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='exp_param', full_name='caffe.LayerParameter.exp_param', index=30,
      number=111, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flatten_param', full_name='caffe.LayerParameter.flatten_param', index=31,
      number=135, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdf5_data_param', full_name='caffe.LayerParameter.hdf5_data_param', index=32,
      number=112, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdf5_output_param', full_name='caffe.LayerParameter.hdf5_output_param', index=33,
      number=113, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hinge_loss_param', full_name='caffe.LayerParameter.hinge_loss_param', index=34,
      number=114, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_data_param', full_name='caffe.LayerParameter.image_data_param', index=35,
      number=115, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='infogain_loss_param', full_name='caffe.LayerParameter.infogain_loss_param', index=36,
      number=116, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inner_product_param', full_name='caffe.LayerParameter.inner_product_param', index=37,
      number=117, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_param', full_name='caffe.LayerParameter.input_param', index=38,
      number=143, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='log_param', full_name='caffe.LayerParameter.log_param', index=39,
      number=134, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lrn_param', full_name='caffe.LayerParameter.lrn_param', index=40,
      number=118, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='memory_data_param', full_name='caffe.LayerParameter.memory_data_param', index=41,
      number=119, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mvn_param', full_name='caffe.LayerParameter.mvn_param', index=42,
      number=120, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pooling_param', full_name='caffe.LayerParameter.pooling_param', index=43,
      number=121, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power_param', full_name='caffe.LayerParameter.power_param', index=44,
      number=122, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prelu_param', full_name='caffe.LayerParameter.prelu_param', index=45,
      number=131, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='python_param', full_name='caffe.LayerParameter.python_param', index=46,
      number=130, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='recurrent_param', full_name='caffe.LayerParameter.recurrent_param', index=47,
      number=146, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reduction_param', full_name='caffe.LayerParameter.reduction_param', index=48,
      number=136, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relu_param', full_name='caffe.LayerParameter.relu_param', index=49,
      number=123, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reshape_param', full_name='caffe.LayerParameter.reshape_param', index=50,
      number=133, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roi_pooling_param', full_name='caffe.LayerParameter.roi_pooling_param', index=51,
      number=8266711, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_param', full_name='caffe.LayerParameter.scale_param', index=52,
      number=142, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sigmoid_param', full_name='caffe.LayerParameter.sigmoid_param', index=53,
      number=124, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='smooth_l1_loss_param', full_name='caffe.LayerParameter.smooth_l1_loss_param', index=54,
      number=8266712, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='softmax_param', full_name='caffe.LayerParameter.softmax_param', index=55,
      number=125, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='spp_param', full_name='caffe.LayerParameter.spp_param', index=56,
      number=132, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slice_param', full_name='caffe.LayerParameter.slice_param', index=57,
      number=126, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tanh_param', full_name='caffe.LayerParameter.tanh_param', index=58,
      number=127, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='threshold_param', full_name='caffe.LayerParameter.threshold_param', index=59,
      number=128, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tile_param', full_name='caffe.LayerParameter.tile_param', index=60,
      number=138, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='window_data_param', full_name='caffe.LayerParameter.window_data_param', index=61,
      number=129, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='st_param', full_name='caffe.LayerParameter.st_param', index=62,
      number=148, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='st_loss_param', full_name='caffe.LayerParameter.st_loss_param', index=63,
      number=145, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn_param', full_name='caffe.LayerParameter.rpn_param', index=64,
      number=150, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focal_loss_param', full_name='caffe.LayerParameter.focal_loss_param', index=65,
      number=155, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='asdn_data_param', full_name='caffe.LayerParameter.asdn_data_param', index=66,
      number=159, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bn_param', full_name='caffe.LayerParameter.bn_param', index=67,
      number=160, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mtcnn_data_param', full_name='caffe.LayerParameter.mtcnn_data_param', index=68,
      number=161, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='interp_param', full_name='caffe.LayerParameter.interp_param', index=69,
      number=162, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='psroi_pooling_param', full_name='caffe.LayerParameter.psroi_pooling_param', index=70,
      number=163, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='annotated_data_param', full_name='caffe.LayerParameter.annotated_data_param', index=71,
      number=164, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prior_box_param', full_name='caffe.LayerParameter.prior_box_param', index=72,
      number=165, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_param', full_name='caffe.LayerParameter.crop_param', index=73,
      number=167, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_evaluate_param', full_name='caffe.LayerParameter.detection_evaluate_param', index=74,
      number=168, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_output_param', full_name='caffe.LayerParameter.detection_output_param', index=75,
      number=169, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='multibox_loss_param', full_name='caffe.LayerParameter.multibox_loss_param', index=76,
      number=171, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='permute_param', full_name='caffe.LayerParameter.permute_param', index=77,
      number=172, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_data_param', full_name='caffe.LayerParameter.video_data_param', index=78,
      number=173, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='margin_inner_product_param', full_name='caffe.LayerParameter.margin_inner_product_param', index=79,
      number=174, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='center_loss_param', full_name='caffe.LayerParameter.center_loss_param', index=80,
      number=175, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deformable_convolution_param', full_name='caffe.LayerParameter.deformable_convolution_param', index=81,
      number=176, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_specific_add_param', full_name='caffe.LayerParameter.label_specific_add_param', index=82,
      number=177, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='additive_margin_inner_product_param', full_name='caffe.LayerParameter.additive_margin_inner_product_param', index=83,
      number=178, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cosin_add_m_param', full_name='caffe.LayerParameter.cosin_add_m_param', index=84,
      number=179, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cosin_mul_m_param', full_name='caffe.LayerParameter.cosin_mul_m_param', index=85,
      number=180, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel_scale_param', full_name='caffe.LayerParameter.channel_scale_param', index=86,
      number=181, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip_param', full_name='caffe.LayerParameter.flip_param', index=87,
      number=182, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='triplet_loss_param', full_name='caffe.LayerParameter.triplet_loss_param', index=88,
      number=183, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coupled_cluster_loss_param', full_name='caffe.LayerParameter.coupled_cluster_loss_param', index=89,
      number=184, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='general_triplet_loss_param', full_name='caffe.LayerParameter.general_triplet_loss_param', index=90,
      number=185, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roi_align_param', full_name='caffe.LayerParameter.roi_align_param', index=91,
      number=186, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upsample_param', full_name='caffe.LayerParameter.upsample_param', index=92,
      number=100003, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='matmul_param', full_name='caffe.LayerParameter.matmul_param', index=93,
      number=100005, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pass_through_param', full_name='caffe.LayerParameter.pass_through_param', index=94,
      number=100004, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='norm_param', full_name='caffe.LayerParameter.norm_param', index=95,
      number=100001, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=4536,
  serialized_end=9293,
)


_UPSAMPLEPARAMETER = _descriptor.Descriptor(
  name='UpsampleParameter',
  full_name='caffe.UpsampleParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.UpsampleParameter.scale', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_h', full_name='caffe.UpsampleParameter.scale_h', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_w', full_name='caffe.UpsampleParameter.scale_w', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_out_h', full_name='caffe.UpsampleParameter.pad_out_h', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_out_w', full_name='caffe.UpsampleParameter.pad_out_w', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upsample_h', full_name='caffe.UpsampleParameter.upsample_h', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upsample_w', full_name='caffe.UpsampleParameter.upsample_w', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9296,
  serialized_end=9459,
)


_MATMULPARAMETER = _descriptor.Descriptor(
  name='MatMulParameter',
  full_name='caffe.MatMulParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dim_1', full_name='caffe.MatMulParameter.dim_1', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dim_2', full_name='caffe.MatMulParameter.dim_2', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dim_3', full_name='caffe.MatMulParameter.dim_3', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9461,
  serialized_end=9523,
)


_PASSTHROUGHPARAMETER = _descriptor.Descriptor(
  name='PassThroughParameter',
  full_name='caffe.PassThroughParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.PassThroughParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block_height', full_name='caffe.PassThroughParameter.block_height', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='block_width', full_name='caffe.PassThroughParameter.block_width', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9525,
  serialized_end=9619,
)


_NORMALIZEPARAMETER = _descriptor.Descriptor(
  name='NormalizeParameter',
  full_name='caffe.NormalizeParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='across_spatial', full_name='caffe.NormalizeParameter.across_spatial', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_filler', full_name='caffe.NormalizeParameter.scale_filler', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel_shared', full_name='caffe.NormalizeParameter.channel_shared', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eps', full_name='caffe.NormalizeParameter.eps', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1e-10),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sqrt_a', full_name='caffe.NormalizeParameter.sqrt_a', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9622,
  serialized_end=9787,
)


_ANNOTATEDDATAPARAMETER = _descriptor.Descriptor(
  name='AnnotatedDataParameter',
  full_name='caffe.AnnotatedDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_sampler', full_name='caffe.AnnotatedDataParameter.batch_sampler', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_map_file', full_name='caffe.AnnotatedDataParameter.label_map_file', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anno_type', full_name='caffe.AnnotatedDataParameter.anno_type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9790,
  serialized_end=9939,
)


_ASDNDATAPARAMETER = _descriptor.Descriptor(
  name='AsdnDataParameter',
  full_name='caffe.AsdnDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='count_drop', full_name='caffe.AsdnDataParameter.count_drop', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=15,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='permute_count', full_name='caffe.AsdnDataParameter.permute_count', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count_drop_neg', full_name='caffe.AsdnDataParameter.count_drop_neg', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channels', full_name='caffe.AsdnDataParameter.channels', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1024,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='iter_size', full_name='caffe.AsdnDataParameter.iter_size', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='maintain_before', full_name='caffe.AsdnDataParameter.maintain_before', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=9942,
  serialized_end=10113,
)


_MTCNNDATAPARAMETER = _descriptor.Descriptor(
  name='MTCNNDataParameter',
  full_name='caffe.MTCNNDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='augmented', full_name='caffe.MTCNNDataParameter.augmented', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip', full_name='caffe.MTCNNDataParameter.flip', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_positive', full_name='caffe.MTCNNDataParameter.num_positive', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_negitive', full_name='caffe.MTCNNDataParameter.num_negitive', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_part', full_name='caffe.MTCNNDataParameter.num_part', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_width', full_name='caffe.MTCNNDataParameter.resize_width', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_height', full_name='caffe.MTCNNDataParameter.resize_height', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_negitive_scale', full_name='caffe.MTCNNDataParameter.min_negitive_scale', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_negitive_scale', full_name='caffe.MTCNNDataParameter.max_negitive_scale', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10116,
  serialized_end=10372,
)


_INTERPPARAMETER = _descriptor.Descriptor(
  name='InterpParameter',
  full_name='caffe.InterpParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='height', full_name='caffe.InterpParameter.height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffe.InterpParameter.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zoom_factor', full_name='caffe.InterpParameter.zoom_factor', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shrink_factor', full_name='caffe.InterpParameter.shrink_factor', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_beg', full_name='caffe.InterpParameter.pad_beg', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_end', full_name='caffe.InterpParameter.pad_end', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10375,
  serialized_end=10519,
)


_PSROIPOOLINGPARAMETER = _descriptor.Descriptor(
  name='PSROIPoolingParameter',
  full_name='caffe.PSROIPoolingParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='spatial_scale', full_name='caffe.PSROIPoolingParameter.spatial_scale', index=0,
      number=1, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_dim', full_name='caffe.PSROIPoolingParameter.output_dim', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group_size', full_name='caffe.PSROIPoolingParameter.group_size', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10521,
  serialized_end=10607,
)


_FLIPPARAMETER = _descriptor.Descriptor(
  name='FlipParameter',
  full_name='caffe.FlipParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='flip_width', full_name='caffe.FlipParameter.flip_width', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip_height', full_name='caffe.FlipParameter.flip_height', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10609,
  serialized_end=10678,
)


_BNPARAMETER = _descriptor.Descriptor(
  name='BNParameter',
  full_name='caffe.BNParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='slope_filler', full_name='caffe.BNParameter.slope_filler', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.BNParameter.bias_filler', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='caffe.BNParameter.momentum', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.9),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eps', full_name='caffe.BNParameter.eps', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1e-05),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frozen', full_name='caffe.BNParameter.frozen', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.BNParameter.engine', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _BNPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10681,
  serialized_end=10948,
)


_FOCALLOSSPARAMETER = _descriptor.Descriptor(
  name='FocalLossParameter',
  full_name='caffe.FocalLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.FocalLossParameter.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gamma', full_name='caffe.FocalLossParameter.gamma', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='caffe.FocalLossParameter.alpha', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.25),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta', full_name='caffe.FocalLossParameter.beta', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _FOCALLOSSPARAMETER_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=10951,
  serialized_end=11113,
)


_TRANSFORMATIONPARAMETER = _descriptor.Descriptor(
  name='TransformationParameter',
  full_name='caffe.TransformationParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.TransformationParameter.scale', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror', full_name='caffe.TransformationParameter.mirror', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_size', full_name='caffe.TransformationParameter.crop_size', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_h', full_name='caffe.TransformationParameter.crop_h', index=3,
      number=11, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_w', full_name='caffe.TransformationParameter.crop_w', index=4,
      number=12, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_file', full_name='caffe.TransformationParameter.mean_file', index=5,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_value', full_name='caffe.TransformationParameter.mean_value', index=6,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_color', full_name='caffe.TransformationParameter.force_color', index=7,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_gray', full_name='caffe.TransformationParameter.force_gray', index=8,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_param', full_name='caffe.TransformationParameter.resize_param', index=9,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='noise_param', full_name='caffe.TransformationParameter.noise_param', index=10,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='distort_param', full_name='caffe.TransformationParameter.distort_param', index=11,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='expand_param', full_name='caffe.TransformationParameter.expand_param', index=12,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='emit_constraint', full_name='caffe.TransformationParameter.emit_constraint', index=13,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=11116,
  serialized_end=11574,
)


_RESIZEPARAMETER = _descriptor.Descriptor(
  name='ResizeParameter',
  full_name='caffe.ResizeParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='prob', full_name='caffe.ResizeParameter.prob', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_mode', full_name='caffe.ResizeParameter.resize_mode', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='caffe.ResizeParameter.height', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffe.ResizeParameter.width', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height_scale', full_name='caffe.ResizeParameter.height_scale', index=4,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width_scale', full_name='caffe.ResizeParameter.width_scale', index=5,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_mode', full_name='caffe.ResizeParameter.pad_mode', index=6,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_value', full_name='caffe.ResizeParameter.pad_value', index=7,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='interp_mode', full_name='caffe.ResizeParameter.interp_mode', index=8,
      number=7, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESIZEPARAMETER_RESIZE_MODE,
    _RESIZEPARAMETER_PAD_MODE,
    _RESIZEPARAMETER_INTERP_MODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=11577,
  serialized_end=12105,
)


_SALTPEPPERPARAMETER = _descriptor.Descriptor(
  name='SaltPepperParameter',
  full_name='caffe.SaltPepperParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='fraction', full_name='caffe.SaltPepperParameter.fraction', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='caffe.SaltPepperParameter.value', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=12107,
  serialized_end=12164,
)


_NOISEPARAMETER = _descriptor.Descriptor(
  name='NoiseParameter',
  full_name='caffe.NoiseParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='prob', full_name='caffe.NoiseParameter.prob', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hist_eq', full_name='caffe.NoiseParameter.hist_eq', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inverse', full_name='caffe.NoiseParameter.inverse', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decolorize', full_name='caffe.NoiseParameter.decolorize', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gauss_blur', full_name='caffe.NoiseParameter.gauss_blur', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='jpeg', full_name='caffe.NoiseParameter.jpeg', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='posterize', full_name='caffe.NoiseParameter.posterize', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='erode', full_name='caffe.NoiseParameter.erode', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saltpepper', full_name='caffe.NoiseParameter.saltpepper', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saltpepper_param', full_name='caffe.NoiseParameter.saltpepper_param', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clahe', full_name='caffe.NoiseParameter.clahe', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='convert_to_hsv', full_name='caffe.NoiseParameter.convert_to_hsv', index=11,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='convert_to_lab', full_name='caffe.NoiseParameter.convert_to_lab', index=12,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=12167,
  serialized_end=12533,
)


_DISTORTIONPARAMETER = _descriptor.Descriptor(
  name='DistortionParameter',
  full_name='caffe.DistortionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='brightness_prob', full_name='caffe.DistortionParameter.brightness_prob', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='brightness_delta', full_name='caffe.DistortionParameter.brightness_delta', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='contrast_prob', full_name='caffe.DistortionParameter.contrast_prob', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='contrast_lower', full_name='caffe.DistortionParameter.contrast_lower', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='contrast_upper', full_name='caffe.DistortionParameter.contrast_upper', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hue_prob', full_name='caffe.DistortionParameter.hue_prob', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hue_delta', full_name='caffe.DistortionParameter.hue_delta', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saturation_prob', full_name='caffe.DistortionParameter.saturation_prob', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saturation_lower', full_name='caffe.DistortionParameter.saturation_lower', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='saturation_upper', full_name='caffe.DistortionParameter.saturation_upper', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_order_prob', full_name='caffe.DistortionParameter.random_order_prob', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=12536,
  serialized_end=12853,
)


_EXPANSIONPARAMETER = _descriptor.Descriptor(
  name='ExpansionParameter',
  full_name='caffe.ExpansionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='prob', full_name='caffe.ExpansionParameter.prob', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_expand_ratio', full_name='caffe.ExpansionParameter.max_expand_ratio', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=12855,
  serialized_end=12921,
)


_LOSSPARAMETER = _descriptor.Descriptor(
  name='LossParameter',
  full_name='caffe.LossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ignore_label', full_name='caffe.LossParameter.ignore_label', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='normalization', full_name='caffe.LossParameter.normalization', index=1,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='normalize', full_name='caffe.LossParameter.normalize', index=2,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LOSSPARAMETER_NORMALIZATIONMODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=12924,
  serialized_end=13118,
)


_ACCURACYPARAMETER = _descriptor.Descriptor(
  name='AccuracyParameter',
  full_name='caffe.AccuracyParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='top_k', full_name='caffe.AccuracyParameter.top_k', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.AccuracyParameter.axis', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ignore_label', full_name='caffe.AccuracyParameter.ignore_label', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13120,
  serialized_end=13196,
)


_ARGMAXPARAMETER = _descriptor.Descriptor(
  name='ArgMaxParameter',
  full_name='caffe.ArgMaxParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='out_max_val', full_name='caffe.ArgMaxParameter.out_max_val', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='top_k', full_name='caffe.ArgMaxParameter.top_k', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.ArgMaxParameter.axis', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13198,
  serialized_end=13275,
)


_CONCATPARAMETER = _descriptor.Descriptor(
  name='ConcatParameter',
  full_name='caffe.ConcatParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.ConcatParameter.axis', index=0,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='concat_dim', full_name='caffe.ConcatParameter.concat_dim', index=1,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13277,
  serialized_end=13334,
)


_BATCHNORMPARAMETER = _descriptor.Descriptor(
  name='BatchNormParameter',
  full_name='caffe.BatchNormParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='use_global_stats', full_name='caffe.BatchNormParameter.use_global_stats', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='moving_average_fraction', full_name='caffe.BatchNormParameter.moving_average_fraction', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.999),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eps', full_name='caffe.BatchNormParameter.eps', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1e-05),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13336,
  serialized_end=13442,
)


_BIASPARAMETER = _descriptor.Descriptor(
  name='BiasParameter',
  full_name='caffe.BiasParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.BiasParameter.axis', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_axes', full_name='caffe.BiasParameter.num_axes', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filler', full_name='caffe.BiasParameter.filler', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13444,
  serialized_end=13537,
)


_CONTRASTIVELOSSPARAMETER = _descriptor.Descriptor(
  name='ContrastiveLossParameter',
  full_name='caffe.ContrastiveLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='margin', full_name='caffe.ContrastiveLossParameter.margin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='legacy_version', full_name='caffe.ContrastiveLossParameter.legacy_version', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13539,
  serialized_end=13615,
)


_DETECTIONLOSSPARAMETER = _descriptor.Descriptor(
  name='DetectionLossParameter',
  full_name='caffe.DetectionLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='side', full_name='caffe.DetectionLossParameter.side', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=7,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_class', full_name='caffe.DetectionLossParameter.num_class', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_object', full_name='caffe.DetectionLossParameter.num_object', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object_scale', full_name='caffe.DetectionLossParameter.object_scale', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='noobject_scale', full_name='caffe.DetectionLossParameter.noobject_scale', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_scale', full_name='caffe.DetectionLossParameter.class_scale', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coord_scale', full_name='caffe.DetectionLossParameter.coord_scale', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sqrt', full_name='caffe.DetectionLossParameter.sqrt', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='constriant', full_name='caffe.DetectionLossParameter.constriant', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13618,
  serialized_end=13854,
)


_REGIONLOSSPARAMETER = _descriptor.Descriptor(
  name='RegionLossParameter',
  full_name='caffe.RegionLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='side', full_name='caffe.RegionLossParameter.side', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=13,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_class', full_name='caffe.RegionLossParameter.num_class', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_match', full_name='caffe.RegionLossParameter.bias_match', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coords', full_name='caffe.RegionLossParameter.coords', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num', full_name='caffe.RegionLossParameter.num', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='softmax', full_name='caffe.RegionLossParameter.softmax', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='jitter', full_name='caffe.RegionLossParameter.jitter', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rescore', full_name='caffe.RegionLossParameter.rescore', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object_scale', full_name='caffe.RegionLossParameter.object_scale', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_scale', full_name='caffe.RegionLossParameter.class_scale', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='noobject_scale', full_name='caffe.RegionLossParameter.noobject_scale', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coord_scale', full_name='caffe.RegionLossParameter.coord_scale', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='absolute', full_name='caffe.RegionLossParameter.absolute', index=12,
      number=13, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='thresh', full_name='caffe.RegionLossParameter.thresh', index=13,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random', full_name='caffe.RegionLossParameter.random', index=14,
      number=15, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='biases', full_name='caffe.RegionLossParameter.biases', index=15,
      number=16, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='softmax_tree', full_name='caffe.RegionLossParameter.softmax_tree', index=16,
      number=17, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_map', full_name='caffe.RegionLossParameter.class_map', index=17,
      number=18, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13857,
  serialized_end=14258,
)


_REORGPARAMETER = _descriptor.Descriptor(
  name='ReorgParameter',
  full_name='caffe.ReorgParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stride', full_name='caffe.ReorgParameter.stride', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reverse', full_name='caffe.ReorgParameter.reverse', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=14260,
  serialized_end=14316,
)


_EVALDETECTIONPARAMETER = _descriptor.Descriptor(
  name='EvalDetectionParameter',
  full_name='caffe.EvalDetectionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='side', full_name='caffe.EvalDetectionParameter.side', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=7,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_class', full_name='caffe.EvalDetectionParameter.num_class', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_object', full_name='caffe.EvalDetectionParameter.num_object', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='threshold', full_name='caffe.EvalDetectionParameter.threshold', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sqrt', full_name='caffe.EvalDetectionParameter.sqrt', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='constriant', full_name='caffe.EvalDetectionParameter.constriant', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score_type', full_name='caffe.EvalDetectionParameter.score_type', index=6,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms', full_name='caffe.EvalDetectionParameter.nms', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='biases', full_name='caffe.EvalDetectionParameter.biases', index=8,
      number=9, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _EVALDETECTIONPARAMETER_SCORETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=14319,
  serialized_end=14626,
)


_CONVOLUTIONPARAMETER = _descriptor.Descriptor(
  name='ConvolutionParameter',
  full_name='caffe.ConvolutionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.ConvolutionParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_term', full_name='caffe.ConvolutionParameter.bias_term', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad', full_name='caffe.ConvolutionParameter.pad', index=2,
      number=3, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_size', full_name='caffe.ConvolutionParameter.kernel_size', index=3,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride', full_name='caffe.ConvolutionParameter.stride', index=4,
      number=6, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dilation', full_name='caffe.ConvolutionParameter.dilation', index=5,
      number=18, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_h', full_name='caffe.ConvolutionParameter.pad_h', index=6,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_w', full_name='caffe.ConvolutionParameter.pad_w', index=7,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_h', full_name='caffe.ConvolutionParameter.kernel_h', index=8,
      number=11, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_w', full_name='caffe.ConvolutionParameter.kernel_w', index=9,
      number=12, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride_h', full_name='caffe.ConvolutionParameter.stride_h', index=10,
      number=13, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride_w', full_name='caffe.ConvolutionParameter.stride_w', index=11,
      number=14, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group', full_name='caffe.ConvolutionParameter.group', index=12,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.ConvolutionParameter.weight_filler', index=13,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.ConvolutionParameter.bias_filler', index=14,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.ConvolutionParameter.engine', index=15,
      number=15, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.ConvolutionParameter.axis', index=16,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_nd_im2col', full_name='caffe.ConvolutionParameter.force_nd_im2col', index=17,
      number=17, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _CONVOLUTIONPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=14629,
  serialized_end=15137,
)


_CROPPARAMETER = _descriptor.Descriptor(
  name='CropParameter',
  full_name='caffe.CropParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.CropParameter.axis', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='offset', full_name='caffe.CropParameter.offset', index=1,
      number=2, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15139,
  serialized_end=15187,
)


_DATAPARAMETER = _descriptor.Descriptor(
  name='DataParameter',
  full_name='caffe.DataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='source', full_name='caffe.DataParameter.source', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='caffe.DataParameter.batch_size', index=1,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rand_skip', full_name='caffe.DataParameter.rand_skip', index=2,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='backend', full_name='caffe.DataParameter.backend', index=3,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.DataParameter.scale', index=4,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_file', full_name='caffe.DataParameter.mean_file', index=5,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_size', full_name='caffe.DataParameter.crop_size', index=6,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror', full_name='caffe.DataParameter.mirror', index=7,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_encoded_color', full_name='caffe.DataParameter.force_encoded_color', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prefetch', full_name='caffe.DataParameter.prefetch', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='side', full_name='caffe.DataParameter.side', index=10,
      number=11, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DATAPARAMETER_DB,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15190,
  serialized_end=15496,
)


_DETECTIONEVALUATEPARAMETER = _descriptor.Descriptor(
  name='DetectionEvaluateParameter',
  full_name='caffe.DetectionEvaluateParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='caffe.DetectionEvaluateParameter.num_classes', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='background_label_id', full_name='caffe.DetectionEvaluateParameter.background_label_id', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='overlap_threshold', full_name='caffe.DetectionEvaluateParameter.overlap_threshold', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='evaluate_difficult_gt', full_name='caffe.DetectionEvaluateParameter.evaluate_difficult_gt', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name_size_file', full_name='caffe.DetectionEvaluateParameter.name_size_file', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_param', full_name='caffe.DetectionEvaluateParameter.resize_param', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15499,
  serialized_end=15719,
)


_NONMAXIMUMSUPPRESSIONPARAMETER = _descriptor.Descriptor(
  name='NonMaximumSuppressionParameter',
  full_name='caffe.NonMaximumSuppressionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nms_threshold', full_name='caffe.NonMaximumSuppressionParameter.nms_threshold', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.3),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='top_k', full_name='caffe.NonMaximumSuppressionParameter.top_k', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eta', full_name='caffe.NonMaximumSuppressionParameter.eta', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15721,
  serialized_end=15812,
)


_SAVEOUTPUTPARAMETER = _descriptor.Descriptor(
  name='SaveOutputParameter',
  full_name='caffe.SaveOutputParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_directory', full_name='caffe.SaveOutputParameter.output_directory', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_name_prefix', full_name='caffe.SaveOutputParameter.output_name_prefix', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_format', full_name='caffe.SaveOutputParameter.output_format', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_map_file', full_name='caffe.SaveOutputParameter.label_map_file', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name_size_file', full_name='caffe.SaveOutputParameter.name_size_file', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_test_image', full_name='caffe.SaveOutputParameter.num_test_image', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_param', full_name='caffe.SaveOutputParameter.resize_param', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15815,
  serialized_end=16031,
)


_DETECTIONOUTPUTPARAMETER = _descriptor.Descriptor(
  name='DetectionOutputParameter',
  full_name='caffe.DetectionOutputParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='caffe.DetectionOutputParameter.num_classes', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='share_location', full_name='caffe.DetectionOutputParameter.share_location', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='background_label_id', full_name='caffe.DetectionOutputParameter.background_label_id', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_param', full_name='caffe.DetectionOutputParameter.nms_param', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_output_param', full_name='caffe.DetectionOutputParameter.save_output_param', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='code_type', full_name='caffe.DetectionOutputParameter.code_type', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='variance_encoded_in_target', full_name='caffe.DetectionOutputParameter.variance_encoded_in_target', index=6,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keep_top_k', full_name='caffe.DetectionOutputParameter.keep_top_k', index=7,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence_threshold', full_name='caffe.DetectionOutputParameter.confidence_threshold', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='visualize', full_name='caffe.DetectionOutputParameter.visualize', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='visualize_threshold', full_name='caffe.DetectionOutputParameter.visualize_threshold', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_file', full_name='caffe.DetectionOutputParameter.save_file', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16034,
  serialized_end=16489,
)


_DROPOUTPARAMETER = _descriptor.Descriptor(
  name='DropoutParameter',
  full_name='caffe.DropoutParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dropout_ratio', full_name='caffe.DropoutParameter.dropout_ratio', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_train', full_name='caffe.DropoutParameter.scale_train', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16491,
  serialized_end=16564,
)


_DUMMYDATAPARAMETER = _descriptor.Descriptor(
  name='DummyDataParameter',
  full_name='caffe.DummyDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_filler', full_name='caffe.DummyDataParameter.data_filler', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='caffe.DummyDataParameter.shape', index=1,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num', full_name='caffe.DummyDataParameter.num', index=2,
      number=2, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channels', full_name='caffe.DummyDataParameter.channels', index=3,
      number=3, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='caffe.DummyDataParameter.height', index=4,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffe.DummyDataParameter.width', index=5,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16567,
  serialized_end=16727,
)


_ELTWISEPARAMETER = _descriptor.Descriptor(
  name='EltwiseParameter',
  full_name='caffe.EltwiseParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='operation', full_name='caffe.EltwiseParameter.operation', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coeff', full_name='caffe.EltwiseParameter.coeff', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stable_prod_grad', full_name='caffe.EltwiseParameter.stable_prod_grad', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ELTWISEPARAMETER_ELTWISEOP,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16730,
  serialized_end=16895,
)


_ELUPARAMETER = _descriptor.Descriptor(
  name='ELUParameter',
  full_name='caffe.ELUParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='alpha', full_name='caffe.ELUParameter.alpha', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16897,
  serialized_end=16929,
)


_EMBEDPARAMETER = _descriptor.Descriptor(
  name='EmbedParameter',
  full_name='caffe.EmbedParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.EmbedParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_dim', full_name='caffe.EmbedParameter.input_dim', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_term', full_name='caffe.EmbedParameter.bias_term', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.EmbedParameter.weight_filler', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.EmbedParameter.bias_filler', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16932,
  serialized_end=17104,
)


_EXPPARAMETER = _descriptor.Descriptor(
  name='ExpParameter',
  full_name='caffe.ExpParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='base', full_name='caffe.ExpParameter.base', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.ExpParameter.scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shift', full_name='caffe.ExpParameter.shift', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17106,
  serialized_end=17174,
)


_FLATTENPARAMETER = _descriptor.Descriptor(
  name='FlattenParameter',
  full_name='caffe.FlattenParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.FlattenParameter.axis', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_axis', full_name='caffe.FlattenParameter.end_axis', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17176,
  serialized_end=17233,
)


_HDF5DATAPARAMETER = _descriptor.Descriptor(
  name='HDF5DataParameter',
  full_name='caffe.HDF5DataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='source', full_name='caffe.HDF5DataParameter.source', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='caffe.HDF5DataParameter.batch_size', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shuffle', full_name='caffe.HDF5DataParameter.shuffle', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17235,
  serialized_end=17314,
)


_HDF5OUTPUTPARAMETER = _descriptor.Descriptor(
  name='HDF5OutputParameter',
  full_name='caffe.HDF5OutputParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='file_name', full_name='caffe.HDF5OutputParameter.file_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17316,
  serialized_end=17356,
)


_HINGELOSSPARAMETER = _descriptor.Descriptor(
  name='HingeLossParameter',
  full_name='caffe.HingeLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='norm', full_name='caffe.HingeLossParameter.norm', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HINGELOSSPARAMETER_NORM,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17358,
  serialized_end=17452,
)


_IMAGEDATAPARAMETER = _descriptor.Descriptor(
  name='ImageDataParameter',
  full_name='caffe.ImageDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='source', full_name='caffe.ImageDataParameter.source', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='caffe.ImageDataParameter.batch_size', index=1,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rand_skip', full_name='caffe.ImageDataParameter.rand_skip', index=2,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shuffle', full_name='caffe.ImageDataParameter.shuffle', index=3,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_height', full_name='caffe.ImageDataParameter.new_height', index=4,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_width', full_name='caffe.ImageDataParameter.new_width', index=5,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_color', full_name='caffe.ImageDataParameter.is_color', index=6,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.ImageDataParameter.scale', index=7,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_file', full_name='caffe.ImageDataParameter.mean_file', index=8,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_size', full_name='caffe.ImageDataParameter.crop_size', index=9,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror', full_name='caffe.ImageDataParameter.mirror', index=10,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='root_folder', full_name='caffe.ImageDataParameter.root_folder', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17455,
  serialized_end=17734,
)


_INFOGAINLOSSPARAMETER = _descriptor.Descriptor(
  name='InfogainLossParameter',
  full_name='caffe.InfogainLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='source', full_name='caffe.InfogainLossParameter.source', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17736,
  serialized_end=17775,
)


_INNERPRODUCTPARAMETER = _descriptor.Descriptor(
  name='InnerProductParameter',
  full_name='caffe.InnerProductParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.InnerProductParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_term', full_name='caffe.InnerProductParameter.bias_term', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.InnerProductParameter.weight_filler', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.InnerProductParameter.bias_filler', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.InnerProductParameter.axis', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transpose', full_name='caffe.InnerProductParameter.transpose', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='normalize', full_name='caffe.InnerProductParameter.normalize', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17778,
  serialized_end=18007,
)


_INPUTPARAMETER = _descriptor.Descriptor(
  name='InputParameter',
  full_name='caffe.InputParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='caffe.InputParameter.shape', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18009,
  serialized_end=18058,
)


_LOGPARAMETER = _descriptor.Descriptor(
  name='LogParameter',
  full_name='caffe.LogParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='base', full_name='caffe.LogParameter.base', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.LogParameter.scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shift', full_name='caffe.LogParameter.shift', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18060,
  serialized_end=18128,
)


_LRNPARAMETER = _descriptor.Descriptor(
  name='LRNParameter',
  full_name='caffe.LRNParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='local_size', full_name='caffe.LRNParameter.local_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='caffe.LRNParameter.alpha', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta', full_name='caffe.LRNParameter.beta', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.75),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='norm_region', full_name='caffe.LRNParameter.norm_region', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='k', full_name='caffe.LRNParameter.k', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.LRNParameter.engine', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LRNPARAMETER_NORMREGION,
    _LRNPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18131,
  serialized_end=18443,
)


_MEMORYDATAPARAMETER = _descriptor.Descriptor(
  name='MemoryDataParameter',
  full_name='caffe.MemoryDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='caffe.MemoryDataParameter.batch_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channels', full_name='caffe.MemoryDataParameter.channels', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='caffe.MemoryDataParameter.height', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='caffe.MemoryDataParameter.width', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18445,
  serialized_end=18535,
)


_MULTIBOXLOSSPARAMETER = _descriptor.Descriptor(
  name='MultiBoxLossParameter',
  full_name='caffe.MultiBoxLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='loc_loss_type', full_name='caffe.MultiBoxLossParameter.loc_loss_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='conf_loss_type', full_name='caffe.MultiBoxLossParameter.conf_loss_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loc_weight', full_name='caffe.MultiBoxLossParameter.loc_weight', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='caffe.MultiBoxLossParameter.num_classes', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='share_location', full_name='caffe.MultiBoxLossParameter.share_location', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='match_type', full_name='caffe.MultiBoxLossParameter.match_type', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='overlap_threshold', full_name='caffe.MultiBoxLossParameter.overlap_threshold', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_prior_for_matching', full_name='caffe.MultiBoxLossParameter.use_prior_for_matching', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='background_label_id', full_name='caffe.MultiBoxLossParameter.background_label_id', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_difficult_gt', full_name='caffe.MultiBoxLossParameter.use_difficult_gt', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='do_neg_mining', full_name='caffe.MultiBoxLossParameter.do_neg_mining', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='neg_pos_ratio', full_name='caffe.MultiBoxLossParameter.neg_pos_ratio', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(3),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='neg_overlap', full_name='caffe.MultiBoxLossParameter.neg_overlap', index=12,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='code_type', full_name='caffe.MultiBoxLossParameter.code_type', index=13,
      number=14, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='encode_variance_in_target', full_name='caffe.MultiBoxLossParameter.encode_variance_in_target', index=14,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='map_object_to_agnostic', full_name='caffe.MultiBoxLossParameter.map_object_to_agnostic', index=15,
      number=17, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ignore_cross_boundary_bbox', full_name='caffe.MultiBoxLossParameter.ignore_cross_boundary_bbox', index=16,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bp_inside', full_name='caffe.MultiBoxLossParameter.bp_inside', index=17,
      number=19, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mining_type', full_name='caffe.MultiBoxLossParameter.mining_type', index=18,
      number=20, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_param', full_name='caffe.MultiBoxLossParameter.nms_param', index=19,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample_size', full_name='caffe.MultiBoxLossParameter.sample_size', index=20,
      number=22, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=64,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_prior_for_nms', full_name='caffe.MultiBoxLossParameter.use_prior_for_nms', index=21,
      number=23, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MULTIBOXLOSSPARAMETER_LOCLOSSTYPE,
    _MULTIBOXLOSSPARAMETER_CONFLOSSTYPE,
    _MULTIBOXLOSSPARAMETER_MATCHTYPE,
    _MULTIBOXLOSSPARAMETER_MININGTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18538,
  serialized_end=19666,
)


_PERMUTEPARAMETER = _descriptor.Descriptor(
  name='PermuteParameter',
  full_name='caffe.PermuteParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='order', full_name='caffe.PermuteParameter.order', index=0,
      number=1, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19668,
  serialized_end=19701,
)


_MVNPARAMETER = _descriptor.Descriptor(
  name='MVNParameter',
  full_name='caffe.MVNParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='normalize_variance', full_name='caffe.MVNParameter.normalize_variance', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='across_channels', full_name='caffe.MVNParameter.across_channels', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eps', full_name='caffe.MVNParameter.eps', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1e-09),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19703,
  serialized_end=19803,
)


_PARAMETERPARAMETER = _descriptor.Descriptor(
  name='ParameterParameter',
  full_name='caffe.ParameterParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='caffe.ParameterParameter.shape', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19805,
  serialized_end=19858,
)


_POOLINGPARAMETER = _descriptor.Descriptor(
  name='PoolingParameter',
  full_name='caffe.PoolingParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pool', full_name='caffe.PoolingParameter.pool', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad', full_name='caffe.PoolingParameter.pad', index=1,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_h', full_name='caffe.PoolingParameter.pad_h', index=2,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_w', full_name='caffe.PoolingParameter.pad_w', index=3,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_size', full_name='caffe.PoolingParameter.kernel_size', index=4,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_h', full_name='caffe.PoolingParameter.kernel_h', index=5,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_w', full_name='caffe.PoolingParameter.kernel_w', index=6,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride', full_name='caffe.PoolingParameter.stride', index=7,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride_h', full_name='caffe.PoolingParameter.stride_h', index=8,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride_w', full_name='caffe.PoolingParameter.stride_w', index=9,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.PoolingParameter.engine', index=10,
      number=11, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_pooling', full_name='caffe.PoolingParameter.global_pooling', index=11,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ceil_mode', full_name='caffe.PoolingParameter.ceil_mode', index=12,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _POOLINGPARAMETER_POOLMETHOD,
    _POOLINGPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19861,
  serialized_end=20304,
)


_POWERPARAMETER = _descriptor.Descriptor(
  name='PowerParameter',
  full_name='caffe.PowerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='power', full_name='caffe.PowerParameter.power', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.PowerParameter.scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shift', full_name='caffe.PowerParameter.shift', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20306,
  serialized_end=20376,
)


_PRIORBOXPARAMETER = _descriptor.Descriptor(
  name='PriorBoxParameter',
  full_name='caffe.PriorBoxParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_size', full_name='caffe.PriorBoxParameter.min_size', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_size', full_name='caffe.PriorBoxParameter.max_size', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='aspect_ratio', full_name='caffe.PriorBoxParameter.aspect_ratio', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip', full_name='caffe.PriorBoxParameter.flip', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clip', full_name='caffe.PriorBoxParameter.clip', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='variance', full_name='caffe.PriorBoxParameter.variance', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='img_size', full_name='caffe.PriorBoxParameter.img_size', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='img_h', full_name='caffe.PriorBoxParameter.img_h', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='img_w', full_name='caffe.PriorBoxParameter.img_w', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='step', full_name='caffe.PriorBoxParameter.step', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='step_h', full_name='caffe.PriorBoxParameter.step_h', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='step_w', full_name='caffe.PriorBoxParameter.step_w', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='offset', full_name='caffe.PriorBoxParameter.offset', index=12,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _PRIORBOXPARAMETER_CODETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20379,
  serialized_end=20688,
)


_PYTHONPARAMETER = _descriptor.Descriptor(
  name='PythonParameter',
  full_name='caffe.PythonParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='module', full_name='caffe.PythonParameter.module', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer', full_name='caffe.PythonParameter.layer', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='param_str', full_name='caffe.PythonParameter.param_str', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='share_in_parallel', full_name='caffe.PythonParameter.share_in_parallel', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20690,
  serialized_end=20793,
)


_RECURRENTPARAMETER = _descriptor.Descriptor(
  name='RecurrentParameter',
  full_name='caffe.RecurrentParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.RecurrentParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.RecurrentParameter.weight_filler', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.RecurrentParameter.bias_filler', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='debug_info', full_name='caffe.RecurrentParameter.debug_info', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='expose_hidden', full_name='caffe.RecurrentParameter.expose_hidden', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20796,
  serialized_end=20988,
)


_REDUCTIONPARAMETER = _descriptor.Descriptor(
  name='ReductionParameter',
  full_name='caffe.ReductionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='operation', full_name='caffe.ReductionParameter.operation', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.ReductionParameter.axis', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coeff', full_name='caffe.ReductionParameter.coeff', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _REDUCTIONPARAMETER_REDUCTIONOP,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20991,
  serialized_end=21164,
)


_RELUPARAMETER = _descriptor.Descriptor(
  name='ReLUParameter',
  full_name='caffe.ReLUParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='negative_slope', full_name='caffe.ReLUParameter.negative_slope', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.ReLUParameter.engine', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RELUPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21167,
  serialized_end=21308,
)


_RESHAPEPARAMETER = _descriptor.Descriptor(
  name='ReshapeParameter',
  full_name='caffe.ReshapeParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='caffe.ReshapeParameter.shape', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.ReshapeParameter.axis', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_axes', full_name='caffe.ReshapeParameter.num_axes', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21310,
  serialized_end=21400,
)


_ROIPOOLINGPARAMETER = _descriptor.Descriptor(
  name='ROIPoolingParameter',
  full_name='caffe.ROIPoolingParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pooled_h', full_name='caffe.ROIPoolingParameter.pooled_h', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pooled_w', full_name='caffe.ROIPoolingParameter.pooled_w', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='spatial_scale', full_name='caffe.ROIPoolingParameter.spatial_scale', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21402,
  serialized_end=21491,
)


_SCALEPARAMETER = _descriptor.Descriptor(
  name='ScaleParameter',
  full_name='caffe.ScaleParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.ScaleParameter.axis', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_axes', full_name='caffe.ScaleParameter.num_axes', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filler', full_name='caffe.ScaleParameter.filler', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_term', full_name='caffe.ScaleParameter.bias_term', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.ScaleParameter.bias_filler', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_value', full_name='caffe.ScaleParameter.min_value', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_value', full_name='caffe.ScaleParameter.max_value', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21494,
  serialized_end=21697,
)


_SIGMOIDPARAMETER = _descriptor.Descriptor(
  name='SigmoidParameter',
  full_name='caffe.SigmoidParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.SigmoidParameter.engine', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SIGMOIDPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21699,
  serialized_end=21819,
)


_SMOOTHL1LOSSPARAMETER = _descriptor.Descriptor(
  name='SmoothL1LossParameter',
  full_name='caffe.SmoothL1LossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sigma', full_name='caffe.SmoothL1LossParameter.sigma', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21821,
  serialized_end=21862,
)


_SLICEPARAMETER = _descriptor.Descriptor(
  name='SliceParameter',
  full_name='caffe.SliceParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.SliceParameter.axis', index=0,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slice_point', full_name='caffe.SliceParameter.slice_point', index=1,
      number=2, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slice_dim', full_name='caffe.SliceParameter.slice_dim', index=2,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21864,
  serialized_end=21940,
)


_SOFTMAXPARAMETER = _descriptor.Descriptor(
  name='SoftmaxParameter',
  full_name='caffe.SoftmaxParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.SoftmaxParameter.engine', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.SoftmaxParameter.axis', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SOFTMAXPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21943,
  serialized_end=22080,
)


_TANHPARAMETER = _descriptor.Descriptor(
  name='TanHParameter',
  full_name='caffe.TanHParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.TanHParameter.engine', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TANHPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22082,
  serialized_end=22196,
)


_TILEPARAMETER = _descriptor.Descriptor(
  name='TileParameter',
  full_name='caffe.TileParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.TileParameter.axis', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tiles', full_name='caffe.TileParameter.tiles', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22198,
  serialized_end=22245,
)


_THRESHOLDPARAMETER = _descriptor.Descriptor(
  name='ThresholdParameter',
  full_name='caffe.ThresholdParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='threshold', full_name='caffe.ThresholdParameter.threshold', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22247,
  serialized_end=22289,
)


_WINDOWDATAPARAMETER = _descriptor.Descriptor(
  name='WindowDataParameter',
  full_name='caffe.WindowDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='source', full_name='caffe.WindowDataParameter.source', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.WindowDataParameter.scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_file', full_name='caffe.WindowDataParameter.mean_file', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='caffe.WindowDataParameter.batch_size', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_size', full_name='caffe.WindowDataParameter.crop_size', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror', full_name='caffe.WindowDataParameter.mirror', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fg_threshold', full_name='caffe.WindowDataParameter.fg_threshold', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bg_threshold', full_name='caffe.WindowDataParameter.bg_threshold', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fg_fraction', full_name='caffe.WindowDataParameter.fg_fraction', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.25),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='context_pad', full_name='caffe.WindowDataParameter.context_pad', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop_mode', full_name='caffe.WindowDataParameter.crop_mode', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("warp").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cache_images', full_name='caffe.WindowDataParameter.cache_images', index=11,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='root_folder', full_name='caffe.WindowDataParameter.root_folder', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22292,
  serialized_end=22613,
)


_SPPPARAMETER = _descriptor.Descriptor(
  name='SPPParameter',
  full_name='caffe.SPPParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pyramid_height', full_name='caffe.SPPParameter.pyramid_height', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pool', full_name='caffe.SPPParameter.pool', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.SPPParameter.engine', index=2,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SPPPARAMETER_POOLMETHOD,
    _SPPPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22616,
  serialized_end=22851,
)


_V1LAYERPARAMETER = _descriptor.Descriptor(
  name='V1LayerParameter',
  full_name='caffe.V1LayerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bottom', full_name='caffe.V1LayerParameter.bottom', index=0,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='top', full_name='caffe.V1LayerParameter.top', index=1,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe.V1LayerParameter.name', index=2,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='include', full_name='caffe.V1LayerParameter.include', index=3,
      number=32, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='exclude', full_name='caffe.V1LayerParameter.exclude', index=4,
      number=33, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.V1LayerParameter.type', index=5,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blobs', full_name='caffe.V1LayerParameter.blobs', index=6,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='param', full_name='caffe.V1LayerParameter.param', index=7,
      number=1001, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blob_share_mode', full_name='caffe.V1LayerParameter.blob_share_mode', index=8,
      number=1002, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blobs_lr', full_name='caffe.V1LayerParameter.blobs_lr', index=9,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='caffe.V1LayerParameter.weight_decay', index=10,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_weight', full_name='caffe.V1LayerParameter.loss_weight', index=11,
      number=35, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='accuracy_param', full_name='caffe.V1LayerParameter.accuracy_param', index=12,
      number=27, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='argmax_param', full_name='caffe.V1LayerParameter.argmax_param', index=13,
      number=23, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='concat_param', full_name='caffe.V1LayerParameter.concat_param', index=14,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='contrastive_loss_param', full_name='caffe.V1LayerParameter.contrastive_loss_param', index=15,
      number=40, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='convolution_param', full_name='caffe.V1LayerParameter.convolution_param', index=16,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_param', full_name='caffe.V1LayerParameter.data_param', index=17,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dropout_param', full_name='caffe.V1LayerParameter.dropout_param', index=18,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dummy_data_param', full_name='caffe.V1LayerParameter.dummy_data_param', index=19,
      number=26, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eltwise_param', full_name='caffe.V1LayerParameter.eltwise_param', index=20,
      number=24, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='exp_param', full_name='caffe.V1LayerParameter.exp_param', index=21,
      number=41, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdf5_data_param', full_name='caffe.V1LayerParameter.hdf5_data_param', index=22,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdf5_output_param', full_name='caffe.V1LayerParameter.hdf5_output_param', index=23,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hinge_loss_param', full_name='caffe.V1LayerParameter.hinge_loss_param', index=24,
      number=29, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_data_param', full_name='caffe.V1LayerParameter.image_data_param', index=25,
      number=15, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='infogain_loss_param', full_name='caffe.V1LayerParameter.infogain_loss_param', index=26,
      number=16, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inner_product_param', full_name='caffe.V1LayerParameter.inner_product_param', index=27,
      number=17, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lrn_param', full_name='caffe.V1LayerParameter.lrn_param', index=28,
      number=18, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='memory_data_param', full_name='caffe.V1LayerParameter.memory_data_param', index=29,
      number=22, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mvn_param', full_name='caffe.V1LayerParameter.mvn_param', index=30,
      number=34, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pooling_param', full_name='caffe.V1LayerParameter.pooling_param', index=31,
      number=19, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power_param', full_name='caffe.V1LayerParameter.power_param', index=32,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relu_param', full_name='caffe.V1LayerParameter.relu_param', index=33,
      number=30, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sigmoid_param', full_name='caffe.V1LayerParameter.sigmoid_param', index=34,
      number=38, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='softmax_param', full_name='caffe.V1LayerParameter.softmax_param', index=35,
      number=39, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slice_param', full_name='caffe.V1LayerParameter.slice_param', index=36,
      number=31, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tanh_param', full_name='caffe.V1LayerParameter.tanh_param', index=37,
      number=37, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='threshold_param', full_name='caffe.V1LayerParameter.threshold_param', index=38,
      number=25, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='window_data_param', full_name='caffe.V1LayerParameter.window_data_param', index=39,
      number=20, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_param', full_name='caffe.V1LayerParameter.transform_param', index=40,
      number=36, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_param', full_name='caffe.V1LayerParameter.loss_param', index=41,
      number=42, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_loss_param', full_name='caffe.V1LayerParameter.detection_loss_param', index=42,
      number=200, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eval_detection_param', full_name='caffe.V1LayerParameter.eval_detection_param', index=43,
      number=201, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer', full_name='caffe.V1LayerParameter.layer', index=44,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _V1LAYERPARAMETER_LAYERTYPE,
    _V1LAYERPARAMETER_DIMCHECKMODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22854,
  serialized_end=25506,
)


_V0LAYERPARAMETER = _descriptor.Descriptor(
  name='V0LayerParameter',
  full_name='caffe.V0LayerParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe.V0LayerParameter.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.V0LayerParameter.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.V0LayerParameter.num_output', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='biasterm', full_name='caffe.V0LayerParameter.biasterm', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.V0LayerParameter.weight_filler', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.V0LayerParameter.bias_filler', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad', full_name='caffe.V0LayerParameter.pad', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernelsize', full_name='caffe.V0LayerParameter.kernelsize', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group', full_name='caffe.V0LayerParameter.group', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride', full_name='caffe.V0LayerParameter.stride', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pool', full_name='caffe.V0LayerParameter.pool', index=10,
      number=11, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dropout_ratio', full_name='caffe.V0LayerParameter.dropout_ratio', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='local_size', full_name='caffe.V0LayerParameter.local_size', index=12,
      number=13, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='caffe.V0LayerParameter.alpha', index=13,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta', full_name='caffe.V0LayerParameter.beta', index=14,
      number=15, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.75),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='k', full_name='caffe.V0LayerParameter.k', index=15,
      number=22, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='source', full_name='caffe.V0LayerParameter.source', index=16,
      number=16, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.V0LayerParameter.scale', index=17,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meanfile', full_name='caffe.V0LayerParameter.meanfile', index=18,
      number=18, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batchsize', full_name='caffe.V0LayerParameter.batchsize', index=19,
      number=19, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cropsize', full_name='caffe.V0LayerParameter.cropsize', index=20,
      number=20, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror', full_name='caffe.V0LayerParameter.mirror', index=21,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blobs', full_name='caffe.V0LayerParameter.blobs', index=22,
      number=50, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blobs_lr', full_name='caffe.V0LayerParameter.blobs_lr', index=23,
      number=51, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='caffe.V0LayerParameter.weight_decay', index=24,
      number=52, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rand_skip', full_name='caffe.V0LayerParameter.rand_skip', index=25,
      number=53, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='det_fg_threshold', full_name='caffe.V0LayerParameter.det_fg_threshold', index=26,
      number=54, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='det_bg_threshold', full_name='caffe.V0LayerParameter.det_bg_threshold', index=27,
      number=55, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='det_fg_fraction', full_name='caffe.V0LayerParameter.det_fg_fraction', index=28,
      number=56, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.25),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='det_context_pad', full_name='caffe.V0LayerParameter.det_context_pad', index=29,
      number=58, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='det_crop_mode', full_name='caffe.V0LayerParameter.det_crop_mode', index=30,
      number=59, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("warp").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_num', full_name='caffe.V0LayerParameter.new_num', index=31,
      number=60, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_channels', full_name='caffe.V0LayerParameter.new_channels', index=32,
      number=61, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_height', full_name='caffe.V0LayerParameter.new_height', index=33,
      number=62, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='new_width', full_name='caffe.V0LayerParameter.new_width', index=34,
      number=63, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shuffle_images', full_name='caffe.V0LayerParameter.shuffle_images', index=35,
      number=64, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='concat_dim', full_name='caffe.V0LayerParameter.concat_dim', index=36,
      number=65, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdf5_output_param', full_name='caffe.V0LayerParameter.hdf5_output_param', index=37,
      number=1001, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _V0LAYERPARAMETER_POOLMETHOD,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=25509,
  serialized_end=26530,
)


_PRELUPARAMETER = _descriptor.Descriptor(
  name='PReLUParameter',
  full_name='caffe.PReLUParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filler', full_name='caffe.PReLUParameter.filler', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel_shared', full_name='caffe.PReLUParameter.channel_shared', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26532,
  serialized_end=26619,
)


_RPNPARAMETER = _descriptor.Descriptor(
  name='RPNParameter',
  full_name='caffe.RPNParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feat_stride', full_name='caffe.RPNParameter.feat_stride', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basesize', full_name='caffe.RPNParameter.basesize', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.RPNParameter.scale', index=2,
      number=3, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ratio', full_name='caffe.RPNParameter.ratio', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='boxminsize', full_name='caffe.RPNParameter.boxminsize', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='per_nms_topn', full_name='caffe.RPNParameter.per_nms_topn', index=5,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='post_nms_topn', full_name='caffe.RPNParameter.post_nms_topn', index=6,
      number=11, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_thresh', full_name='caffe.RPNParameter.nms_thresh', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26622,
  serialized_end=26790,
)


_VIDEODATAPARAMETER = _descriptor.Descriptor(
  name='VideoDataParameter',
  full_name='caffe.VideoDataParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='video_type', full_name='caffe.VideoDataParameter.video_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_id', full_name='caffe.VideoDataParameter.device_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_file', full_name='caffe.VideoDataParameter.video_file', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='skip_frames', full_name='caffe.VideoDataParameter.skip_frames', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _VIDEODATAPARAMETER_VIDEOTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26793,
  serialized_end=26980,
)


_CENTERLOSSPARAMETER = _descriptor.Descriptor(
  name='CenterLossParameter',
  full_name='caffe.CenterLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.CenterLossParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='center_filler', full_name='caffe.CenterLossParameter.center_filler', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.CenterLossParameter.axis', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26982,
  serialized_end=27087,
)


_MARGININNERPRODUCTPARAMETER = _descriptor.Descriptor(
  name='MarginInnerProductParameter',
  full_name='caffe.MarginInnerProductParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.MarginInnerProductParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='caffe.MarginInnerProductParameter.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.MarginInnerProductParameter.weight_filler', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.MarginInnerProductParameter.axis', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='base', full_name='caffe.MarginInnerProductParameter.base', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gamma', full_name='caffe.MarginInnerProductParameter.gamma', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power', full_name='caffe.MarginInnerProductParameter.power', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='iteration', full_name='caffe.MarginInnerProductParameter.iteration', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lambda_min', full_name='caffe.MarginInnerProductParameter.lambda_min', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MARGININNERPRODUCTPARAMETER_MARGINTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27090,
  serialized_end=27435,
)


_ADDITIVEMARGININNERPRODUCTPARAMETER = _descriptor.Descriptor(
  name='AdditiveMarginInnerProductParameter',
  full_name='caffe.AdditiveMarginInnerProductParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.AdditiveMarginInnerProductParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.AdditiveMarginInnerProductParameter.weight_filler', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='m', full_name='caffe.AdditiveMarginInnerProductParameter.m', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.35),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.AdditiveMarginInnerProductParameter.axis', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27438,
  serialized_end=27576,
)


_DEFORMABLECONVOLUTIONPARAMETER = _descriptor.Descriptor(
  name='DeformableConvolutionParameter',
  full_name='caffe.DeformableConvolutionParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_output', full_name='caffe.DeformableConvolutionParameter.num_output', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_term', full_name='caffe.DeformableConvolutionParameter.bias_term', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad', full_name='caffe.DeformableConvolutionParameter.pad', index=2,
      number=3, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_size', full_name='caffe.DeformableConvolutionParameter.kernel_size', index=3,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride', full_name='caffe.DeformableConvolutionParameter.stride', index=4,
      number=6, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dilation', full_name='caffe.DeformableConvolutionParameter.dilation', index=5,
      number=18, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_h', full_name='caffe.DeformableConvolutionParameter.pad_h', index=6,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_w', full_name='caffe.DeformableConvolutionParameter.pad_w', index=7,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_h', full_name='caffe.DeformableConvolutionParameter.kernel_h', index=8,
      number=11, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_w', full_name='caffe.DeformableConvolutionParameter.kernel_w', index=9,
      number=12, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride_h', full_name='caffe.DeformableConvolutionParameter.stride_h', index=10,
      number=13, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride_w', full_name='caffe.DeformableConvolutionParameter.stride_w', index=11,
      number=14, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group', full_name='caffe.DeformableConvolutionParameter.group', index=12,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deformable_group', full_name='caffe.DeformableConvolutionParameter.deformable_group', index=13,
      number=25, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_filler', full_name='caffe.DeformableConvolutionParameter.weight_filler', index=14,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias_filler', full_name='caffe.DeformableConvolutionParameter.bias_filler', index=15,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine', full_name='caffe.DeformableConvolutionParameter.engine', index=16,
      number=15, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='caffe.DeformableConvolutionParameter.axis', index=17,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_nd_im2col', full_name='caffe.DeformableConvolutionParameter.force_nd_im2col', index=18,
      number=17, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DEFORMABLECONVOLUTIONPARAMETER_ENGINE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27579,
  serialized_end=28136,
)


_LABELSPECIFICADDPARAMETER = _descriptor.Descriptor(
  name='LabelSpecificAddParameter',
  full_name='caffe.LabelSpecificAddParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bias', full_name='caffe.LabelSpecificAddParameter.bias', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_test', full_name='caffe.LabelSpecificAddParameter.transform_test', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28138,
  serialized_end=28213,
)


_CHANNELSCALEPARAMETER = _descriptor.Descriptor(
  name='ChannelScaleParameter',
  full_name='caffe.ChannelScaleParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='do_forward', full_name='caffe.ChannelScaleParameter.do_forward', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='do_backward_feature', full_name='caffe.ChannelScaleParameter.do_backward_feature', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='do_backward_scale', full_name='caffe.ChannelScaleParameter.do_backward_scale', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_scale', full_name='caffe.ChannelScaleParameter.global_scale', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_global_scale', full_name='caffe.ChannelScaleParameter.max_global_scale', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1000),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_global_scale', full_name='caffe.ChannelScaleParameter.min_global_scale', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='init_global_scale', full_name='caffe.ChannelScaleParameter.init_global_scale', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28216,
  serialized_end=28453,
)


_COSINADDMPARAMETER = _descriptor.Descriptor(
  name='CosinAddmParameter',
  full_name='caffe.CosinAddmParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='m', full_name='caffe.CosinAddmParameter.m', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_test', full_name='caffe.CosinAddmParameter.transform_test', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28455,
  serialized_end=28522,
)


_COSINMULMPARAMETER = _descriptor.Descriptor(
  name='CosinMulmParameter',
  full_name='caffe.CosinMulmParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='m', full_name='caffe.CosinMulmParameter.m', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(4),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transform_test', full_name='caffe.CosinMulmParameter.transform_test', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28524,
  serialized_end=28589,
)


_COUPLEDCLUSTERLOSSPARAMETER = _descriptor.Descriptor(
  name='CoupledClusterLossParameter',
  full_name='caffe.CoupledClusterLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='margin', full_name='caffe.CoupledClusterLossParameter.margin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group_size', full_name='caffe.CoupledClusterLossParameter.group_size', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.CoupledClusterLossParameter.scale', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='log_flag', full_name='caffe.CoupledClusterLossParameter.log_flag', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28591,
  serialized_end=28705,
)


_TRIPLETLOSSPARAMETER = _descriptor.Descriptor(
  name='TripletLossParameter',
  full_name='caffe.TripletLossParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='margin', full_name='caffe.TripletLossParameter.margin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group_size', full_name='caffe.TripletLossParameter.group_size', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='caffe.TripletLossParameter.scale', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28707,
  serialized_end=28789,
)


_GENERALTRIPLETPARAMETER = _descriptor.Descriptor(
  name='GeneralTripletParameter',
  full_name='caffe.GeneralTripletParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='margin', full_name='caffe.GeneralTripletParameter.margin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='add_center_loss', full_name='caffe.GeneralTripletParameter.add_center_loss', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hardest_only', full_name='caffe.GeneralTripletParameter.hardest_only', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='positive_first', full_name='caffe.GeneralTripletParameter.positive_first', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='positive_upper_bound', full_name='caffe.GeneralTripletParameter.positive_upper_bound', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='positive_weight', full_name='caffe.GeneralTripletParameter.positive_weight', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='negative_weight', full_name='caffe.GeneralTripletParameter.negative_weight', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28792,
  serialized_end=29018,
)


_ROIALIGNPARAMETER = _descriptor.Descriptor(
  name='ROIAlignParameter',
  full_name='caffe.ROIAlignParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pooled_h', full_name='caffe.ROIAlignParameter.pooled_h', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pooled_w', full_name='caffe.ROIAlignParameter.pooled_w', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='spatial_scale', full_name='caffe.ROIAlignParameter.spatial_scale', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29020,
  serialized_end=29107,
)

_BLOBPROTO.fields_by_name['shape'].message_type = _BLOBSHAPE
_BLOBPROTOVECTOR.fields_by_name['blobs'].message_type = _BLOBPROTO
_LABELMAP.fields_by_name['item'].message_type = _LABELMAPITEM
_BATCHSAMPLER.fields_by_name['sampler'].message_type = _SAMPLER
_BATCHSAMPLER.fields_by_name['sample_constraint'].message_type = _SAMPLECONSTRAINT
_EMITCONSTRAINT.fields_by_name['emit_type'].enum_type = _EMITCONSTRAINT_EMITTYPE
_EMITCONSTRAINT_EMITTYPE.containing_type = _EMITCONSTRAINT
_ANNOTATION.fields_by_name['bbox'].message_type = _NORMALIZEDBBOX
_ANNOTATIONGROUP.fields_by_name['annotation'].message_type = _ANNOTATION
_ANNOTATEDDATUM.fields_by_name['datum'].message_type = _DATUM
_ANNOTATEDDATUM.fields_by_name['type'].enum_type = _ANNOTATEDDATUM_ANNOTATIONTYPE
_ANNOTATEDDATUM.fields_by_name['annotation_group'].message_type = _ANNOTATIONGROUP
_ANNOTATEDDATUM_ANNOTATIONTYPE.containing_type = _ANNOTATEDDATUM
_MTCNNDATUM.fields_by_name['datum'].message_type = _DATUM
_MTCNNDATUM.fields_by_name['roi'].message_type = _MTCNNBBOX
_FILLERPARAMETER.fields_by_name['variance_norm'].enum_type = _FILLERPARAMETER_VARIANCENORM
_FILLERPARAMETER_VARIANCENORM.containing_type = _FILLERPARAMETER
_NETPARAMETER.fields_by_name['input_shape'].message_type = _BLOBSHAPE
_NETPARAMETER.fields_by_name['state'].message_type = _NETSTATE
_NETPARAMETER.fields_by_name['layer'].message_type = _LAYERPARAMETER
_NETPARAMETER.fields_by_name['layers'].message_type = _V1LAYERPARAMETER
_SOLVERPARAMETER.fields_by_name['net_param'].message_type = _NETPARAMETER
_SOLVERPARAMETER.fields_by_name['train_net_param'].message_type = _NETPARAMETER
_SOLVERPARAMETER.fields_by_name['test_net_param'].message_type = _NETPARAMETER
_SOLVERPARAMETER.fields_by_name['train_state'].message_type = _NETSTATE
_SOLVERPARAMETER.fields_by_name['test_state'].message_type = _NETSTATE
_SOLVERPARAMETER.fields_by_name['snapshot_format'].enum_type = _SOLVERPARAMETER_SNAPSHOTFORMAT
_SOLVERPARAMETER.fields_by_name['solver_mode'].enum_type = _SOLVERPARAMETER_SOLVERMODE
_SOLVERPARAMETER.fields_by_name['solver_type'].enum_type = _SOLVERPARAMETER_SOLVERTYPE
_SOLVERPARAMETER_SNAPSHOTFORMAT.containing_type = _SOLVERPARAMETER
_SOLVERPARAMETER_SOLVERMODE.containing_type = _SOLVERPARAMETER
_SOLVERPARAMETER_SOLVERTYPE.containing_type = _SOLVERPARAMETER
_SOLVERSTATE.fields_by_name['history'].message_type = _BLOBPROTO
_NETSTATE.fields_by_name['phase'].enum_type = _PHASE
_NETSTATERULE.fields_by_name['phase'].enum_type = _PHASE
_PARAMSPEC.fields_by_name['share_mode'].enum_type = _PARAMSPEC_DIMCHECKMODE
_PARAMSPEC_DIMCHECKMODE.containing_type = _PARAMSPEC
_LAYERPARAMETER.fields_by_name['phase'].enum_type = _PHASE
_LAYERPARAMETER.fields_by_name['param'].message_type = _PARAMSPEC
_LAYERPARAMETER.fields_by_name['blobs'].message_type = _BLOBPROTO
_LAYERPARAMETER.fields_by_name['include'].message_type = _NETSTATERULE
_LAYERPARAMETER.fields_by_name['exclude'].message_type = _NETSTATERULE
_LAYERPARAMETER.fields_by_name['transform_param'].message_type = _TRANSFORMATIONPARAMETER
_LAYERPARAMETER.fields_by_name['loss_param'].message_type = _LOSSPARAMETER
_LAYERPARAMETER.fields_by_name['detection_loss_param'].message_type = _DETECTIONLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['eval_detection_param'].message_type = _EVALDETECTIONPARAMETER
_LAYERPARAMETER.fields_by_name['region_loss_param'].message_type = _REGIONLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['reorg_param'].message_type = _REORGPARAMETER
_LAYERPARAMETER.fields_by_name['accuracy_param'].message_type = _ACCURACYPARAMETER
_LAYERPARAMETER.fields_by_name['argmax_param'].message_type = _ARGMAXPARAMETER
_LAYERPARAMETER.fields_by_name['batch_norm_param'].message_type = _BATCHNORMPARAMETER
_LAYERPARAMETER.fields_by_name['bias_param'].message_type = _BIASPARAMETER
_LAYERPARAMETER.fields_by_name['concat_param'].message_type = _CONCATPARAMETER
_LAYERPARAMETER.fields_by_name['contrastive_loss_param'].message_type = _CONTRASTIVELOSSPARAMETER
_LAYERPARAMETER.fields_by_name['convolution_param'].message_type = _CONVOLUTIONPARAMETER
_LAYERPARAMETER.fields_by_name['data_param'].message_type = _DATAPARAMETER
_LAYERPARAMETER.fields_by_name['dropout_param'].message_type = _DROPOUTPARAMETER
_LAYERPARAMETER.fields_by_name['dummy_data_param'].message_type = _DUMMYDATAPARAMETER
_LAYERPARAMETER.fields_by_name['eltwise_param'].message_type = _ELTWISEPARAMETER
_LAYERPARAMETER.fields_by_name['elu_param'].message_type = _ELUPARAMETER
_LAYERPARAMETER.fields_by_name['embed_param'].message_type = _EMBEDPARAMETER
_LAYERPARAMETER.fields_by_name['exp_param'].message_type = _EXPPARAMETER
_LAYERPARAMETER.fields_by_name['flatten_param'].message_type = _FLATTENPARAMETER
_LAYERPARAMETER.fields_by_name['hdf5_data_param'].message_type = _HDF5DATAPARAMETER
_LAYERPARAMETER.fields_by_name['hdf5_output_param'].message_type = _HDF5OUTPUTPARAMETER
_LAYERPARAMETER.fields_by_name['hinge_loss_param'].message_type = _HINGELOSSPARAMETER
_LAYERPARAMETER.fields_by_name['image_data_param'].message_type = _IMAGEDATAPARAMETER
_LAYERPARAMETER.fields_by_name['infogain_loss_param'].message_type = _INFOGAINLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['inner_product_param'].message_type = _INNERPRODUCTPARAMETER
_LAYERPARAMETER.fields_by_name['input_param'].message_type = _INPUTPARAMETER
_LAYERPARAMETER.fields_by_name['log_param'].message_type = _LOGPARAMETER
_LAYERPARAMETER.fields_by_name['lrn_param'].message_type = _LRNPARAMETER
_LAYERPARAMETER.fields_by_name['memory_data_param'].message_type = _MEMORYDATAPARAMETER
_LAYERPARAMETER.fields_by_name['mvn_param'].message_type = _MVNPARAMETER
_LAYERPARAMETER.fields_by_name['pooling_param'].message_type = _POOLINGPARAMETER
_LAYERPARAMETER.fields_by_name['power_param'].message_type = _POWERPARAMETER
_LAYERPARAMETER.fields_by_name['prelu_param'].message_type = _PRELUPARAMETER
_LAYERPARAMETER.fields_by_name['python_param'].message_type = _PYTHONPARAMETER
_LAYERPARAMETER.fields_by_name['recurrent_param'].message_type = _RECURRENTPARAMETER
_LAYERPARAMETER.fields_by_name['reduction_param'].message_type = _REDUCTIONPARAMETER
_LAYERPARAMETER.fields_by_name['relu_param'].message_type = _RELUPARAMETER
_LAYERPARAMETER.fields_by_name['reshape_param'].message_type = _RESHAPEPARAMETER
_LAYERPARAMETER.fields_by_name['roi_pooling_param'].message_type = _ROIPOOLINGPARAMETER
_LAYERPARAMETER.fields_by_name['scale_param'].message_type = _SCALEPARAMETER
_LAYERPARAMETER.fields_by_name['sigmoid_param'].message_type = _SIGMOIDPARAMETER
_LAYERPARAMETER.fields_by_name['smooth_l1_loss_param'].message_type = _SMOOTHL1LOSSPARAMETER
_LAYERPARAMETER.fields_by_name['softmax_param'].message_type = _SOFTMAXPARAMETER
_LAYERPARAMETER.fields_by_name['spp_param'].message_type = _SPPPARAMETER
_LAYERPARAMETER.fields_by_name['slice_param'].message_type = _SLICEPARAMETER
_LAYERPARAMETER.fields_by_name['tanh_param'].message_type = _TANHPARAMETER
_LAYERPARAMETER.fields_by_name['threshold_param'].message_type = _THRESHOLDPARAMETER
_LAYERPARAMETER.fields_by_name['tile_param'].message_type = _TILEPARAMETER
_LAYERPARAMETER.fields_by_name['window_data_param'].message_type = _WINDOWDATAPARAMETER
_LAYERPARAMETER.fields_by_name['st_param'].message_type = _SPATIALTRANSFORMERPARAMETER
_LAYERPARAMETER.fields_by_name['st_loss_param'].message_type = _STLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['rpn_param'].message_type = _RPNPARAMETER
_LAYERPARAMETER.fields_by_name['focal_loss_param'].message_type = _FOCALLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['asdn_data_param'].message_type = _ASDNDATAPARAMETER
_LAYERPARAMETER.fields_by_name['bn_param'].message_type = _BNPARAMETER
_LAYERPARAMETER.fields_by_name['mtcnn_data_param'].message_type = _MTCNNDATAPARAMETER
_LAYERPARAMETER.fields_by_name['interp_param'].message_type = _INTERPPARAMETER
_LAYERPARAMETER.fields_by_name['psroi_pooling_param'].message_type = _PSROIPOOLINGPARAMETER
_LAYERPARAMETER.fields_by_name['annotated_data_param'].message_type = _ANNOTATEDDATAPARAMETER
_LAYERPARAMETER.fields_by_name['prior_box_param'].message_type = _PRIORBOXPARAMETER
_LAYERPARAMETER.fields_by_name['crop_param'].message_type = _CROPPARAMETER
_LAYERPARAMETER.fields_by_name['detection_evaluate_param'].message_type = _DETECTIONEVALUATEPARAMETER
_LAYERPARAMETER.fields_by_name['detection_output_param'].message_type = _DETECTIONOUTPUTPARAMETER
_LAYERPARAMETER.fields_by_name['multibox_loss_param'].message_type = _MULTIBOXLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['permute_param'].message_type = _PERMUTEPARAMETER
_LAYERPARAMETER.fields_by_name['video_data_param'].message_type = _VIDEODATAPARAMETER
_LAYERPARAMETER.fields_by_name['margin_inner_product_param'].message_type = _MARGININNERPRODUCTPARAMETER
_LAYERPARAMETER.fields_by_name['center_loss_param'].message_type = _CENTERLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['deformable_convolution_param'].message_type = _DEFORMABLECONVOLUTIONPARAMETER
_LAYERPARAMETER.fields_by_name['label_specific_add_param'].message_type = _LABELSPECIFICADDPARAMETER
_LAYERPARAMETER.fields_by_name['additive_margin_inner_product_param'].message_type = _ADDITIVEMARGININNERPRODUCTPARAMETER
_LAYERPARAMETER.fields_by_name['cosin_add_m_param'].message_type = _COSINADDMPARAMETER
_LAYERPARAMETER.fields_by_name['cosin_mul_m_param'].message_type = _COSINMULMPARAMETER
_LAYERPARAMETER.fields_by_name['channel_scale_param'].message_type = _CHANNELSCALEPARAMETER
_LAYERPARAMETER.fields_by_name['flip_param'].message_type = _FLIPPARAMETER
_LAYERPARAMETER.fields_by_name['triplet_loss_param'].message_type = _TRIPLETLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['coupled_cluster_loss_param'].message_type = _COUPLEDCLUSTERLOSSPARAMETER
_LAYERPARAMETER.fields_by_name['general_triplet_loss_param'].message_type = _GENERALTRIPLETPARAMETER
_LAYERPARAMETER.fields_by_name['roi_align_param'].message_type = _ROIALIGNPARAMETER
_LAYERPARAMETER.fields_by_name['upsample_param'].message_type = _UPSAMPLEPARAMETER
_LAYERPARAMETER.fields_by_name['matmul_param'].message_type = _MATMULPARAMETER
_LAYERPARAMETER.fields_by_name['pass_through_param'].message_type = _PASSTHROUGHPARAMETER
_LAYERPARAMETER.fields_by_name['norm_param'].message_type = _NORMALIZEPARAMETER
_NORMALIZEPARAMETER.fields_by_name['scale_filler'].message_type = _FILLERPARAMETER
_ANNOTATEDDATAPARAMETER.fields_by_name['batch_sampler'].message_type = _BATCHSAMPLER
_ANNOTATEDDATAPARAMETER.fields_by_name['anno_type'].enum_type = _ANNOTATEDDATUM_ANNOTATIONTYPE
_BNPARAMETER.fields_by_name['slope_filler'].message_type = _FILLERPARAMETER
_BNPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_BNPARAMETER.fields_by_name['engine'].enum_type = _BNPARAMETER_ENGINE
_BNPARAMETER_ENGINE.containing_type = _BNPARAMETER
_FOCALLOSSPARAMETER.fields_by_name['type'].enum_type = _FOCALLOSSPARAMETER_TYPE
_FOCALLOSSPARAMETER_TYPE.containing_type = _FOCALLOSSPARAMETER
_TRANSFORMATIONPARAMETER.fields_by_name['resize_param'].message_type = _RESIZEPARAMETER
_TRANSFORMATIONPARAMETER.fields_by_name['noise_param'].message_type = _NOISEPARAMETER
_TRANSFORMATIONPARAMETER.fields_by_name['distort_param'].message_type = _DISTORTIONPARAMETER
_TRANSFORMATIONPARAMETER.fields_by_name['expand_param'].message_type = _EXPANSIONPARAMETER
_TRANSFORMATIONPARAMETER.fields_by_name['emit_constraint'].message_type = _EMITCONSTRAINT
_RESIZEPARAMETER.fields_by_name['resize_mode'].enum_type = _RESIZEPARAMETER_RESIZE_MODE
_RESIZEPARAMETER.fields_by_name['pad_mode'].enum_type = _RESIZEPARAMETER_PAD_MODE
_RESIZEPARAMETER.fields_by_name['interp_mode'].enum_type = _RESIZEPARAMETER_INTERP_MODE
_RESIZEPARAMETER_RESIZE_MODE.containing_type = _RESIZEPARAMETER
_RESIZEPARAMETER_PAD_MODE.containing_type = _RESIZEPARAMETER
_RESIZEPARAMETER_INTERP_MODE.containing_type = _RESIZEPARAMETER
_NOISEPARAMETER.fields_by_name['saltpepper_param'].message_type = _SALTPEPPERPARAMETER
_LOSSPARAMETER.fields_by_name['normalization'].enum_type = _LOSSPARAMETER_NORMALIZATIONMODE
_LOSSPARAMETER_NORMALIZATIONMODE.containing_type = _LOSSPARAMETER
_BIASPARAMETER.fields_by_name['filler'].message_type = _FILLERPARAMETER
_EVALDETECTIONPARAMETER.fields_by_name['score_type'].enum_type = _EVALDETECTIONPARAMETER_SCORETYPE
_EVALDETECTIONPARAMETER_SCORETYPE.containing_type = _EVALDETECTIONPARAMETER
_CONVOLUTIONPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_CONVOLUTIONPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_CONVOLUTIONPARAMETER.fields_by_name['engine'].enum_type = _CONVOLUTIONPARAMETER_ENGINE
_CONVOLUTIONPARAMETER_ENGINE.containing_type = _CONVOLUTIONPARAMETER
_DATAPARAMETER.fields_by_name['backend'].enum_type = _DATAPARAMETER_DB
_DATAPARAMETER_DB.containing_type = _DATAPARAMETER
_DETECTIONEVALUATEPARAMETER.fields_by_name['resize_param'].message_type = _RESIZEPARAMETER
_SAVEOUTPUTPARAMETER.fields_by_name['resize_param'].message_type = _RESIZEPARAMETER
_DETECTIONOUTPUTPARAMETER.fields_by_name['nms_param'].message_type = _NONMAXIMUMSUPPRESSIONPARAMETER
_DETECTIONOUTPUTPARAMETER.fields_by_name['save_output_param'].message_type = _SAVEOUTPUTPARAMETER
_DETECTIONOUTPUTPARAMETER.fields_by_name['code_type'].enum_type = _PRIORBOXPARAMETER_CODETYPE
_DUMMYDATAPARAMETER.fields_by_name['data_filler'].message_type = _FILLERPARAMETER
_DUMMYDATAPARAMETER.fields_by_name['shape'].message_type = _BLOBSHAPE
_ELTWISEPARAMETER.fields_by_name['operation'].enum_type = _ELTWISEPARAMETER_ELTWISEOP
_ELTWISEPARAMETER_ELTWISEOP.containing_type = _ELTWISEPARAMETER
_EMBEDPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_EMBEDPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_HINGELOSSPARAMETER.fields_by_name['norm'].enum_type = _HINGELOSSPARAMETER_NORM
_HINGELOSSPARAMETER_NORM.containing_type = _HINGELOSSPARAMETER
_INNERPRODUCTPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_INNERPRODUCTPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_INPUTPARAMETER.fields_by_name['shape'].message_type = _BLOBSHAPE
_LRNPARAMETER.fields_by_name['norm_region'].enum_type = _LRNPARAMETER_NORMREGION
_LRNPARAMETER.fields_by_name['engine'].enum_type = _LRNPARAMETER_ENGINE
_LRNPARAMETER_NORMREGION.containing_type = _LRNPARAMETER
_LRNPARAMETER_ENGINE.containing_type = _LRNPARAMETER
_MULTIBOXLOSSPARAMETER.fields_by_name['loc_loss_type'].enum_type = _MULTIBOXLOSSPARAMETER_LOCLOSSTYPE
_MULTIBOXLOSSPARAMETER.fields_by_name['conf_loss_type'].enum_type = _MULTIBOXLOSSPARAMETER_CONFLOSSTYPE
_MULTIBOXLOSSPARAMETER.fields_by_name['match_type'].enum_type = _MULTIBOXLOSSPARAMETER_MATCHTYPE
_MULTIBOXLOSSPARAMETER.fields_by_name['code_type'].enum_type = _PRIORBOXPARAMETER_CODETYPE
_MULTIBOXLOSSPARAMETER.fields_by_name['mining_type'].enum_type = _MULTIBOXLOSSPARAMETER_MININGTYPE
_MULTIBOXLOSSPARAMETER.fields_by_name['nms_param'].message_type = _NONMAXIMUMSUPPRESSIONPARAMETER
_MULTIBOXLOSSPARAMETER_LOCLOSSTYPE.containing_type = _MULTIBOXLOSSPARAMETER
_MULTIBOXLOSSPARAMETER_CONFLOSSTYPE.containing_type = _MULTIBOXLOSSPARAMETER
_MULTIBOXLOSSPARAMETER_MATCHTYPE.containing_type = _MULTIBOXLOSSPARAMETER
_MULTIBOXLOSSPARAMETER_MININGTYPE.containing_type = _MULTIBOXLOSSPARAMETER
_PARAMETERPARAMETER.fields_by_name['shape'].message_type = _BLOBSHAPE
_POOLINGPARAMETER.fields_by_name['pool'].enum_type = _POOLINGPARAMETER_POOLMETHOD
_POOLINGPARAMETER.fields_by_name['engine'].enum_type = _POOLINGPARAMETER_ENGINE
_POOLINGPARAMETER_POOLMETHOD.containing_type = _POOLINGPARAMETER
_POOLINGPARAMETER_ENGINE.containing_type = _POOLINGPARAMETER
_PRIORBOXPARAMETER_CODETYPE.containing_type = _PRIORBOXPARAMETER
_RECURRENTPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_RECURRENTPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_REDUCTIONPARAMETER.fields_by_name['operation'].enum_type = _REDUCTIONPARAMETER_REDUCTIONOP
_REDUCTIONPARAMETER_REDUCTIONOP.containing_type = _REDUCTIONPARAMETER
_RELUPARAMETER.fields_by_name['engine'].enum_type = _RELUPARAMETER_ENGINE
_RELUPARAMETER_ENGINE.containing_type = _RELUPARAMETER
_RESHAPEPARAMETER.fields_by_name['shape'].message_type = _BLOBSHAPE
_SCALEPARAMETER.fields_by_name['filler'].message_type = _FILLERPARAMETER
_SCALEPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_SIGMOIDPARAMETER.fields_by_name['engine'].enum_type = _SIGMOIDPARAMETER_ENGINE
_SIGMOIDPARAMETER_ENGINE.containing_type = _SIGMOIDPARAMETER
_SOFTMAXPARAMETER.fields_by_name['engine'].enum_type = _SOFTMAXPARAMETER_ENGINE
_SOFTMAXPARAMETER_ENGINE.containing_type = _SOFTMAXPARAMETER
_TANHPARAMETER.fields_by_name['engine'].enum_type = _TANHPARAMETER_ENGINE
_TANHPARAMETER_ENGINE.containing_type = _TANHPARAMETER
_SPPPARAMETER.fields_by_name['pool'].enum_type = _SPPPARAMETER_POOLMETHOD
_SPPPARAMETER.fields_by_name['engine'].enum_type = _SPPPARAMETER_ENGINE
_SPPPARAMETER_POOLMETHOD.containing_type = _SPPPARAMETER
_SPPPARAMETER_ENGINE.containing_type = _SPPPARAMETER
_V1LAYERPARAMETER.fields_by_name['include'].message_type = _NETSTATERULE
_V1LAYERPARAMETER.fields_by_name['exclude'].message_type = _NETSTATERULE
_V1LAYERPARAMETER.fields_by_name['type'].enum_type = _V1LAYERPARAMETER_LAYERTYPE
_V1LAYERPARAMETER.fields_by_name['blobs'].message_type = _BLOBPROTO
_V1LAYERPARAMETER.fields_by_name['blob_share_mode'].enum_type = _V1LAYERPARAMETER_DIMCHECKMODE
_V1LAYERPARAMETER.fields_by_name['accuracy_param'].message_type = _ACCURACYPARAMETER
_V1LAYERPARAMETER.fields_by_name['argmax_param'].message_type = _ARGMAXPARAMETER
_V1LAYERPARAMETER.fields_by_name['concat_param'].message_type = _CONCATPARAMETER
_V1LAYERPARAMETER.fields_by_name['contrastive_loss_param'].message_type = _CONTRASTIVELOSSPARAMETER
_V1LAYERPARAMETER.fields_by_name['convolution_param'].message_type = _CONVOLUTIONPARAMETER
_V1LAYERPARAMETER.fields_by_name['data_param'].message_type = _DATAPARAMETER
_V1LAYERPARAMETER.fields_by_name['dropout_param'].message_type = _DROPOUTPARAMETER
_V1LAYERPARAMETER.fields_by_name['dummy_data_param'].message_type = _DUMMYDATAPARAMETER
_V1LAYERPARAMETER.fields_by_name['eltwise_param'].message_type = _ELTWISEPARAMETER
_V1LAYERPARAMETER.fields_by_name['exp_param'].message_type = _EXPPARAMETER
_V1LAYERPARAMETER.fields_by_name['hdf5_data_param'].message_type = _HDF5DATAPARAMETER
_V1LAYERPARAMETER.fields_by_name['hdf5_output_param'].message_type = _HDF5OUTPUTPARAMETER
_V1LAYERPARAMETER.fields_by_name['hinge_loss_param'].message_type = _HINGELOSSPARAMETER
_V1LAYERPARAMETER.fields_by_name['image_data_param'].message_type = _IMAGEDATAPARAMETER
_V1LAYERPARAMETER.fields_by_name['infogain_loss_param'].message_type = _INFOGAINLOSSPARAMETER
_V1LAYERPARAMETER.fields_by_name['inner_product_param'].message_type = _INNERPRODUCTPARAMETER
_V1LAYERPARAMETER.fields_by_name['lrn_param'].message_type = _LRNPARAMETER
_V1LAYERPARAMETER.fields_by_name['memory_data_param'].message_type = _MEMORYDATAPARAMETER
_V1LAYERPARAMETER.fields_by_name['mvn_param'].message_type = _MVNPARAMETER
_V1LAYERPARAMETER.fields_by_name['pooling_param'].message_type = _POOLINGPARAMETER
_V1LAYERPARAMETER.fields_by_name['power_param'].message_type = _POWERPARAMETER
_V1LAYERPARAMETER.fields_by_name['relu_param'].message_type = _RELUPARAMETER
_V1LAYERPARAMETER.fields_by_name['sigmoid_param'].message_type = _SIGMOIDPARAMETER
_V1LAYERPARAMETER.fields_by_name['softmax_param'].message_type = _SOFTMAXPARAMETER
_V1LAYERPARAMETER.fields_by_name['slice_param'].message_type = _SLICEPARAMETER
_V1LAYERPARAMETER.fields_by_name['tanh_param'].message_type = _TANHPARAMETER
_V1LAYERPARAMETER.fields_by_name['threshold_param'].message_type = _THRESHOLDPARAMETER
_V1LAYERPARAMETER.fields_by_name['window_data_param'].message_type = _WINDOWDATAPARAMETER
_V1LAYERPARAMETER.fields_by_name['transform_param'].message_type = _TRANSFORMATIONPARAMETER
_V1LAYERPARAMETER.fields_by_name['loss_param'].message_type = _LOSSPARAMETER
_V1LAYERPARAMETER.fields_by_name['detection_loss_param'].message_type = _DETECTIONLOSSPARAMETER
_V1LAYERPARAMETER.fields_by_name['eval_detection_param'].message_type = _EVALDETECTIONPARAMETER
_V1LAYERPARAMETER.fields_by_name['layer'].message_type = _V0LAYERPARAMETER
_V1LAYERPARAMETER_LAYERTYPE.containing_type = _V1LAYERPARAMETER
_V1LAYERPARAMETER_DIMCHECKMODE.containing_type = _V1LAYERPARAMETER
_V0LAYERPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_V0LAYERPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_V0LAYERPARAMETER.fields_by_name['pool'].enum_type = _V0LAYERPARAMETER_POOLMETHOD
_V0LAYERPARAMETER.fields_by_name['blobs'].message_type = _BLOBPROTO
_V0LAYERPARAMETER.fields_by_name['hdf5_output_param'].message_type = _HDF5OUTPUTPARAMETER
_V0LAYERPARAMETER_POOLMETHOD.containing_type = _V0LAYERPARAMETER
_PRELUPARAMETER.fields_by_name['filler'].message_type = _FILLERPARAMETER
_VIDEODATAPARAMETER.fields_by_name['video_type'].enum_type = _VIDEODATAPARAMETER_VIDEOTYPE
_VIDEODATAPARAMETER_VIDEOTYPE.containing_type = _VIDEODATAPARAMETER
_CENTERLOSSPARAMETER.fields_by_name['center_filler'].message_type = _FILLERPARAMETER
_MARGININNERPRODUCTPARAMETER.fields_by_name['type'].enum_type = _MARGININNERPRODUCTPARAMETER_MARGINTYPE
_MARGININNERPRODUCTPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_MARGININNERPRODUCTPARAMETER_MARGINTYPE.containing_type = _MARGININNERPRODUCTPARAMETER
_ADDITIVEMARGININNERPRODUCTPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_DEFORMABLECONVOLUTIONPARAMETER.fields_by_name['weight_filler'].message_type = _FILLERPARAMETER
_DEFORMABLECONVOLUTIONPARAMETER.fields_by_name['bias_filler'].message_type = _FILLERPARAMETER
_DEFORMABLECONVOLUTIONPARAMETER.fields_by_name['engine'].enum_type = _DEFORMABLECONVOLUTIONPARAMETER_ENGINE
_DEFORMABLECONVOLUTIONPARAMETER_ENGINE.containing_type = _DEFORMABLECONVOLUTIONPARAMETER
DESCRIPTOR.message_types_by_name['BlobShape'] = _BLOBSHAPE
DESCRIPTOR.message_types_by_name['BlobProto'] = _BLOBPROTO
DESCRIPTOR.message_types_by_name['BlobProtoVector'] = _BLOBPROTOVECTOR
DESCRIPTOR.message_types_by_name['Datum'] = _DATUM
DESCRIPTOR.message_types_by_name['LabelMapItem'] = _LABELMAPITEM
DESCRIPTOR.message_types_by_name['LabelMap'] = _LABELMAP
DESCRIPTOR.message_types_by_name['Sampler'] = _SAMPLER
DESCRIPTOR.message_types_by_name['SampleConstraint'] = _SAMPLECONSTRAINT
DESCRIPTOR.message_types_by_name['BatchSampler'] = _BATCHSAMPLER
DESCRIPTOR.message_types_by_name['EmitConstraint'] = _EMITCONSTRAINT
DESCRIPTOR.message_types_by_name['NormalizedBBox'] = _NORMALIZEDBBOX
DESCRIPTOR.message_types_by_name['Annotation'] = _ANNOTATION
DESCRIPTOR.message_types_by_name['AnnotationGroup'] = _ANNOTATIONGROUP
DESCRIPTOR.message_types_by_name['AnnotatedDatum'] = _ANNOTATEDDATUM
DESCRIPTOR.message_types_by_name['MTCNNBBox'] = _MTCNNBBOX
DESCRIPTOR.message_types_by_name['MTCNNDatum'] = _MTCNNDATUM
DESCRIPTOR.message_types_by_name['FillerParameter'] = _FILLERPARAMETER
DESCRIPTOR.message_types_by_name['NetParameter'] = _NETPARAMETER
DESCRIPTOR.message_types_by_name['SolverParameter'] = _SOLVERPARAMETER
DESCRIPTOR.message_types_by_name['SolverState'] = _SOLVERSTATE
DESCRIPTOR.message_types_by_name['NetState'] = _NETSTATE
DESCRIPTOR.message_types_by_name['NetStateRule'] = _NETSTATERULE
DESCRIPTOR.message_types_by_name['SpatialTransformerParameter'] = _SPATIALTRANSFORMERPARAMETER
DESCRIPTOR.message_types_by_name['STLossParameter'] = _STLOSSPARAMETER
DESCRIPTOR.message_types_by_name['ParamSpec'] = _PARAMSPEC
DESCRIPTOR.message_types_by_name['LayerParameter'] = _LAYERPARAMETER
DESCRIPTOR.message_types_by_name['UpsampleParameter'] = _UPSAMPLEPARAMETER
DESCRIPTOR.message_types_by_name['MatMulParameter'] = _MATMULPARAMETER
DESCRIPTOR.message_types_by_name['PassThroughParameter'] = _PASSTHROUGHPARAMETER
DESCRIPTOR.message_types_by_name['NormalizeParameter'] = _NORMALIZEPARAMETER
DESCRIPTOR.message_types_by_name['AnnotatedDataParameter'] = _ANNOTATEDDATAPARAMETER
DESCRIPTOR.message_types_by_name['AsdnDataParameter'] = _ASDNDATAPARAMETER
DESCRIPTOR.message_types_by_name['MTCNNDataParameter'] = _MTCNNDATAPARAMETER
DESCRIPTOR.message_types_by_name['InterpParameter'] = _INTERPPARAMETER
DESCRIPTOR.message_types_by_name['PSROIPoolingParameter'] = _PSROIPOOLINGPARAMETER
DESCRIPTOR.message_types_by_name['FlipParameter'] = _FLIPPARAMETER
DESCRIPTOR.message_types_by_name['BNParameter'] = _BNPARAMETER
DESCRIPTOR.message_types_by_name['FocalLossParameter'] = _FOCALLOSSPARAMETER
DESCRIPTOR.message_types_by_name['TransformationParameter'] = _TRANSFORMATIONPARAMETER
DESCRIPTOR.message_types_by_name['ResizeParameter'] = _RESIZEPARAMETER
DESCRIPTOR.message_types_by_name['SaltPepperParameter'] = _SALTPEPPERPARAMETER
DESCRIPTOR.message_types_by_name['NoiseParameter'] = _NOISEPARAMETER
DESCRIPTOR.message_types_by_name['DistortionParameter'] = _DISTORTIONPARAMETER
DESCRIPTOR.message_types_by_name['ExpansionParameter'] = _EXPANSIONPARAMETER
DESCRIPTOR.message_types_by_name['LossParameter'] = _LOSSPARAMETER
DESCRIPTOR.message_types_by_name['AccuracyParameter'] = _ACCURACYPARAMETER
DESCRIPTOR.message_types_by_name['ArgMaxParameter'] = _ARGMAXPARAMETER
DESCRIPTOR.message_types_by_name['ConcatParameter'] = _CONCATPARAMETER
DESCRIPTOR.message_types_by_name['BatchNormParameter'] = _BATCHNORMPARAMETER
DESCRIPTOR.message_types_by_name['BiasParameter'] = _BIASPARAMETER
DESCRIPTOR.message_types_by_name['ContrastiveLossParameter'] = _CONTRASTIVELOSSPARAMETER
DESCRIPTOR.message_types_by_name['DetectionLossParameter'] = _DETECTIONLOSSPARAMETER
DESCRIPTOR.message_types_by_name['RegionLossParameter'] = _REGIONLOSSPARAMETER
DESCRIPTOR.message_types_by_name['ReorgParameter'] = _REORGPARAMETER
DESCRIPTOR.message_types_by_name['EvalDetectionParameter'] = _EVALDETECTIONPARAMETER
DESCRIPTOR.message_types_by_name['ConvolutionParameter'] = _CONVOLUTIONPARAMETER
DESCRIPTOR.message_types_by_name['CropParameter'] = _CROPPARAMETER
DESCRIPTOR.message_types_by_name['DataParameter'] = _DATAPARAMETER
DESCRIPTOR.message_types_by_name['DetectionEvaluateParameter'] = _DETECTIONEVALUATEPARAMETER
DESCRIPTOR.message_types_by_name['NonMaximumSuppressionParameter'] = _NONMAXIMUMSUPPRESSIONPARAMETER
DESCRIPTOR.message_types_by_name['SaveOutputParameter'] = _SAVEOUTPUTPARAMETER
DESCRIPTOR.message_types_by_name['DetectionOutputParameter'] = _DETECTIONOUTPUTPARAMETER
DESCRIPTOR.message_types_by_name['DropoutParameter'] = _DROPOUTPARAMETER
DESCRIPTOR.message_types_by_name['DummyDataParameter'] = _DUMMYDATAPARAMETER
DESCRIPTOR.message_types_by_name['EltwiseParameter'] = _ELTWISEPARAMETER
DESCRIPTOR.message_types_by_name['ELUParameter'] = _ELUPARAMETER
DESCRIPTOR.message_types_by_name['EmbedParameter'] = _EMBEDPARAMETER
DESCRIPTOR.message_types_by_name['ExpParameter'] = _EXPPARAMETER
DESCRIPTOR.message_types_by_name['FlattenParameter'] = _FLATTENPARAMETER
DESCRIPTOR.message_types_by_name['HDF5DataParameter'] = _HDF5DATAPARAMETER
DESCRIPTOR.message_types_by_name['HDF5OutputParameter'] = _HDF5OUTPUTPARAMETER
DESCRIPTOR.message_types_by_name['HingeLossParameter'] = _HINGELOSSPARAMETER
DESCRIPTOR.message_types_by_name['ImageDataParameter'] = _IMAGEDATAPARAMETER
DESCRIPTOR.message_types_by_name['InfogainLossParameter'] = _INFOGAINLOSSPARAMETER
DESCRIPTOR.message_types_by_name['InnerProductParameter'] = _INNERPRODUCTPARAMETER
DESCRIPTOR.message_types_by_name['InputParameter'] = _INPUTPARAMETER
DESCRIPTOR.message_types_by_name['LogParameter'] = _LOGPARAMETER
DESCRIPTOR.message_types_by_name['LRNParameter'] = _LRNPARAMETER
DESCRIPTOR.message_types_by_name['MemoryDataParameter'] = _MEMORYDATAPARAMETER
DESCRIPTOR.message_types_by_name['MultiBoxLossParameter'] = _MULTIBOXLOSSPARAMETER
DESCRIPTOR.message_types_by_name['PermuteParameter'] = _PERMUTEPARAMETER
DESCRIPTOR.message_types_by_name['MVNParameter'] = _MVNPARAMETER
DESCRIPTOR.message_types_by_name['ParameterParameter'] = _PARAMETERPARAMETER
DESCRIPTOR.message_types_by_name['PoolingParameter'] = _POOLINGPARAMETER
DESCRIPTOR.message_types_by_name['PowerParameter'] = _POWERPARAMETER
DESCRIPTOR.message_types_by_name['PriorBoxParameter'] = _PRIORBOXPARAMETER
DESCRIPTOR.message_types_by_name['PythonParameter'] = _PYTHONPARAMETER
DESCRIPTOR.message_types_by_name['RecurrentParameter'] = _RECURRENTPARAMETER
DESCRIPTOR.message_types_by_name['ReductionParameter'] = _REDUCTIONPARAMETER
DESCRIPTOR.message_types_by_name['ReLUParameter'] = _RELUPARAMETER
DESCRIPTOR.message_types_by_name['ReshapeParameter'] = _RESHAPEPARAMETER
DESCRIPTOR.message_types_by_name['ROIPoolingParameter'] = _ROIPOOLINGPARAMETER
DESCRIPTOR.message_types_by_name['ScaleParameter'] = _SCALEPARAMETER
DESCRIPTOR.message_types_by_name['SigmoidParameter'] = _SIGMOIDPARAMETER
DESCRIPTOR.message_types_by_name['SmoothL1LossParameter'] = _SMOOTHL1LOSSPARAMETER
DESCRIPTOR.message_types_by_name['SliceParameter'] = _SLICEPARAMETER
DESCRIPTOR.message_types_by_name['SoftmaxParameter'] = _SOFTMAXPARAMETER
DESCRIPTOR.message_types_by_name['TanHParameter'] = _TANHPARAMETER
DESCRIPTOR.message_types_by_name['TileParameter'] = _TILEPARAMETER
DESCRIPTOR.message_types_by_name['ThresholdParameter'] = _THRESHOLDPARAMETER
DESCRIPTOR.message_types_by_name['WindowDataParameter'] = _WINDOWDATAPARAMETER
DESCRIPTOR.message_types_by_name['SPPParameter'] = _SPPPARAMETER
DESCRIPTOR.message_types_by_name['V1LayerParameter'] = _V1LAYERPARAMETER
DESCRIPTOR.message_types_by_name['V0LayerParameter'] = _V0LAYERPARAMETER
DESCRIPTOR.message_types_by_name['PReLUParameter'] = _PRELUPARAMETER
DESCRIPTOR.message_types_by_name['RPNParameter'] = _RPNPARAMETER
DESCRIPTOR.message_types_by_name['VideoDataParameter'] = _VIDEODATAPARAMETER
DESCRIPTOR.message_types_by_name['CenterLossParameter'] = _CENTERLOSSPARAMETER
DESCRIPTOR.message_types_by_name['MarginInnerProductParameter'] = _MARGININNERPRODUCTPARAMETER
DESCRIPTOR.message_types_by_name['AdditiveMarginInnerProductParameter'] = _ADDITIVEMARGININNERPRODUCTPARAMETER
DESCRIPTOR.message_types_by_name['DeformableConvolutionParameter'] = _DEFORMABLECONVOLUTIONPARAMETER
DESCRIPTOR.message_types_by_name['LabelSpecificAddParameter'] = _LABELSPECIFICADDPARAMETER
DESCRIPTOR.message_types_by_name['ChannelScaleParameter'] = _CHANNELSCALEPARAMETER
DESCRIPTOR.message_types_by_name['CosinAddmParameter'] = _COSINADDMPARAMETER
DESCRIPTOR.message_types_by_name['CosinMulmParameter'] = _COSINMULMPARAMETER
DESCRIPTOR.message_types_by_name['CoupledClusterLossParameter'] = _COUPLEDCLUSTERLOSSPARAMETER
DESCRIPTOR.message_types_by_name['TripletLossParameter'] = _TRIPLETLOSSPARAMETER
DESCRIPTOR.message_types_by_name['GeneralTripletParameter'] = _GENERALTRIPLETPARAMETER
DESCRIPTOR.message_types_by_name['ROIAlignParameter'] = _ROIALIGNPARAMETER
DESCRIPTOR.enum_types_by_name['Phase'] = _PHASE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BlobShape = _reflection.GeneratedProtocolMessageType('BlobShape', (_message.Message,), dict(
  DESCRIPTOR = _BLOBSHAPE,
  __module__ = 'caffe_pb2'

  ))
_sym_db.RegisterMessage(BlobShape)

BlobProto = _reflection.GeneratedProtocolMessageType('BlobProto', (_message.Message,), dict(
  DESCRIPTOR = _BLOBPROTO,
  __module__ = 'caffe_pb2'

  ))
_sym_db.RegisterMessage(BlobProto)

BlobProtoVector = _reflection.GeneratedProtocolMessageType('BlobProtoVector', (_message.Message,), dict(
  DESCRIPTOR = _BLOBPROTOVECTOR,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(BlobProtoVector)

Datum = _reflection.GeneratedProtocolMessageType('Datum', (_message.Message,), dict(
  DESCRIPTOR = _DATUM,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(Datum)

LabelMapItem = _reflection.GeneratedProtocolMessageType('LabelMapItem', (_message.Message,), dict(
  DESCRIPTOR = _LABELMAPITEM,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(LabelMapItem)

LabelMap = _reflection.GeneratedProtocolMessageType('LabelMap', (_message.Message,), dict(
  DESCRIPTOR = _LABELMAP,
  __module__ = 'caffe_pb2'
 
  ))
_sym_db.RegisterMessage(LabelMap)

Sampler = _reflection.GeneratedProtocolMessageType('Sampler', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLER,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(Sampler)

SampleConstraint = _reflection.GeneratedProtocolMessageType('SampleConstraint', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLECONSTRAINT,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(SampleConstraint)

BatchSampler = _reflection.GeneratedProtocolMessageType('BatchSampler', (_message.Message,), dict(
  DESCRIPTOR = _BATCHSAMPLER,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(BatchSampler)

EmitConstraint = _reflection.GeneratedProtocolMessageType('EmitConstraint', (_message.Message,), dict(
  DESCRIPTOR = _EMITCONSTRAINT,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(EmitConstraint)

NormalizedBBox = _reflection.GeneratedProtocolMessageType('NormalizedBBox', (_message.Message,), dict(
  DESCRIPTOR = _NORMALIZEDBBOX,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(NormalizedBBox)

Annotation = _reflection.GeneratedProtocolMessageType('Annotation', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATION,
  __module__ = 'caffe_pb2'
  
  ))
_sym_db.RegisterMessage(Annotation)

AnnotationGroup = _reflection.GeneratedProtocolMessageType('AnnotationGroup', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATIONGROUP,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.AnnotationGroup)
  ))
_sym_db.RegisterMessage(AnnotationGroup)

AnnotatedDatum = _reflection.GeneratedProtocolMessageType('AnnotatedDatum', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATEDDATUM,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.AnnotatedDatum)
  ))
_sym_db.RegisterMessage(AnnotatedDatum)

MTCNNBBox = _reflection.GeneratedProtocolMessageType('MTCNNBBox', (_message.Message,), dict(
  DESCRIPTOR = _MTCNNBBOX,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MTCNNBBox)
  ))
_sym_db.RegisterMessage(MTCNNBBox)

MTCNNDatum = _reflection.GeneratedProtocolMessageType('MTCNNDatum', (_message.Message,), dict(
  DESCRIPTOR = _MTCNNDATUM,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MTCNNDatum)
  ))
_sym_db.RegisterMessage(MTCNNDatum)

FillerParameter = _reflection.GeneratedProtocolMessageType('FillerParameter', (_message.Message,), dict(
  DESCRIPTOR = _FILLERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.FillerParameter)
  ))
_sym_db.RegisterMessage(FillerParameter)

NetParameter = _reflection.GeneratedProtocolMessageType('NetParameter', (_message.Message,), dict(
  DESCRIPTOR = _NETPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.NetParameter)
  ))
_sym_db.RegisterMessage(NetParameter)

SolverParameter = _reflection.GeneratedProtocolMessageType('SolverParameter', (_message.Message,), dict(
  DESCRIPTOR = _SOLVERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SolverParameter)
  ))
_sym_db.RegisterMessage(SolverParameter)

SolverState = _reflection.GeneratedProtocolMessageType('SolverState', (_message.Message,), dict(
  DESCRIPTOR = _SOLVERSTATE,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SolverState)
  ))
_sym_db.RegisterMessage(SolverState)

NetState = _reflection.GeneratedProtocolMessageType('NetState', (_message.Message,), dict(
  DESCRIPTOR = _NETSTATE,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.NetState)
  ))
_sym_db.RegisterMessage(NetState)

NetStateRule = _reflection.GeneratedProtocolMessageType('NetStateRule', (_message.Message,), dict(
  DESCRIPTOR = _NETSTATERULE,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.NetStateRule)
  ))
_sym_db.RegisterMessage(NetStateRule)

SpatialTransformerParameter = _reflection.GeneratedProtocolMessageType('SpatialTransformerParameter', (_message.Message,), dict(
  DESCRIPTOR = _SPATIALTRANSFORMERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SpatialTransformerParameter)
  ))
_sym_db.RegisterMessage(SpatialTransformerParameter)

STLossParameter = _reflection.GeneratedProtocolMessageType('STLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _STLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.STLossParameter)
  ))
_sym_db.RegisterMessage(STLossParameter)

ParamSpec = _reflection.GeneratedProtocolMessageType('ParamSpec', (_message.Message,), dict(
  DESCRIPTOR = _PARAMSPEC,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ParamSpec)
  ))
_sym_db.RegisterMessage(ParamSpec)

LayerParameter = _reflection.GeneratedProtocolMessageType('LayerParameter', (_message.Message,), dict(
  DESCRIPTOR = _LAYERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.LayerParameter)
  ))
_sym_db.RegisterMessage(LayerParameter)

UpsampleParameter = _reflection.GeneratedProtocolMessageType('UpsampleParameter', (_message.Message,), dict(
  DESCRIPTOR = _UPSAMPLEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.UpsampleParameter)
  ))
_sym_db.RegisterMessage(UpsampleParameter)

MatMulParameter = _reflection.GeneratedProtocolMessageType('MatMulParameter', (_message.Message,), dict(
  DESCRIPTOR = _MATMULPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MatMulParameter)
  ))
_sym_db.RegisterMessage(MatMulParameter)

PassThroughParameter = _reflection.GeneratedProtocolMessageType('PassThroughParameter', (_message.Message,), dict(
  DESCRIPTOR = _PASSTHROUGHPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PassThroughParameter)
  ))
_sym_db.RegisterMessage(PassThroughParameter)

NormalizeParameter = _reflection.GeneratedProtocolMessageType('NormalizeParameter', (_message.Message,), dict(
  DESCRIPTOR = _NORMALIZEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.NormalizeParameter)
  ))
_sym_db.RegisterMessage(NormalizeParameter)

AnnotatedDataParameter = _reflection.GeneratedProtocolMessageType('AnnotatedDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATEDDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.AnnotatedDataParameter)
  ))
_sym_db.RegisterMessage(AnnotatedDataParameter)

AsdnDataParameter = _reflection.GeneratedProtocolMessageType('AsdnDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _ASDNDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.AsdnDataParameter)
  ))
_sym_db.RegisterMessage(AsdnDataParameter)

MTCNNDataParameter = _reflection.GeneratedProtocolMessageType('MTCNNDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _MTCNNDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MTCNNDataParameter)
  ))
_sym_db.RegisterMessage(MTCNNDataParameter)

InterpParameter = _reflection.GeneratedProtocolMessageType('InterpParameter', (_message.Message,), dict(
  DESCRIPTOR = _INTERPPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.InterpParameter)
  ))
_sym_db.RegisterMessage(InterpParameter)

PSROIPoolingParameter = _reflection.GeneratedProtocolMessageType('PSROIPoolingParameter', (_message.Message,), dict(
  DESCRIPTOR = _PSROIPOOLINGPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PSROIPoolingParameter)
  ))
_sym_db.RegisterMessage(PSROIPoolingParameter)

FlipParameter = _reflection.GeneratedProtocolMessageType('FlipParameter', (_message.Message,), dict(
  DESCRIPTOR = _FLIPPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.FlipParameter)
  ))
_sym_db.RegisterMessage(FlipParameter)

BNParameter = _reflection.GeneratedProtocolMessageType('BNParameter', (_message.Message,), dict(
  DESCRIPTOR = _BNPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.BNParameter)
  ))
_sym_db.RegisterMessage(BNParameter)

FocalLossParameter = _reflection.GeneratedProtocolMessageType('FocalLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _FOCALLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.FocalLossParameter)
  ))
_sym_db.RegisterMessage(FocalLossParameter)

TransformationParameter = _reflection.GeneratedProtocolMessageType('TransformationParameter', (_message.Message,), dict(
  DESCRIPTOR = _TRANSFORMATIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.TransformationParameter)
  ))
_sym_db.RegisterMessage(TransformationParameter)

ResizeParameter = _reflection.GeneratedProtocolMessageType('ResizeParameter', (_message.Message,), dict(
  DESCRIPTOR = _RESIZEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ResizeParameter)
  ))
_sym_db.RegisterMessage(ResizeParameter)

SaltPepperParameter = _reflection.GeneratedProtocolMessageType('SaltPepperParameter', (_message.Message,), dict(
  DESCRIPTOR = _SALTPEPPERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SaltPepperParameter)
  ))
_sym_db.RegisterMessage(SaltPepperParameter)

NoiseParameter = _reflection.GeneratedProtocolMessageType('NoiseParameter', (_message.Message,), dict(
  DESCRIPTOR = _NOISEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.NoiseParameter)
  ))
_sym_db.RegisterMessage(NoiseParameter)

DistortionParameter = _reflection.GeneratedProtocolMessageType('DistortionParameter', (_message.Message,), dict(
  DESCRIPTOR = _DISTORTIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DistortionParameter)
  ))
_sym_db.RegisterMessage(DistortionParameter)

ExpansionParameter = _reflection.GeneratedProtocolMessageType('ExpansionParameter', (_message.Message,), dict(
  DESCRIPTOR = _EXPANSIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ExpansionParameter)
  ))
_sym_db.RegisterMessage(ExpansionParameter)

LossParameter = _reflection.GeneratedProtocolMessageType('LossParameter', (_message.Message,), dict(
  DESCRIPTOR = _LOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.LossParameter)
  ))
_sym_db.RegisterMessage(LossParameter)

AccuracyParameter = _reflection.GeneratedProtocolMessageType('AccuracyParameter', (_message.Message,), dict(
  DESCRIPTOR = _ACCURACYPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.AccuracyParameter)
  ))
_sym_db.RegisterMessage(AccuracyParameter)

ArgMaxParameter = _reflection.GeneratedProtocolMessageType('ArgMaxParameter', (_message.Message,), dict(
  DESCRIPTOR = _ARGMAXPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ArgMaxParameter)
  ))
_sym_db.RegisterMessage(ArgMaxParameter)

ConcatParameter = _reflection.GeneratedProtocolMessageType('ConcatParameter', (_message.Message,), dict(
  DESCRIPTOR = _CONCATPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ConcatParameter)
  ))
_sym_db.RegisterMessage(ConcatParameter)

BatchNormParameter = _reflection.GeneratedProtocolMessageType('BatchNormParameter', (_message.Message,), dict(
  DESCRIPTOR = _BATCHNORMPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.BatchNormParameter)
  ))
_sym_db.RegisterMessage(BatchNormParameter)

BiasParameter = _reflection.GeneratedProtocolMessageType('BiasParameter', (_message.Message,), dict(
  DESCRIPTOR = _BIASPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.BiasParameter)
  ))
_sym_db.RegisterMessage(BiasParameter)

ContrastiveLossParameter = _reflection.GeneratedProtocolMessageType('ContrastiveLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _CONTRASTIVELOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ContrastiveLossParameter)
  ))
_sym_db.RegisterMessage(ContrastiveLossParameter)

DetectionLossParameter = _reflection.GeneratedProtocolMessageType('DetectionLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _DETECTIONLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DetectionLossParameter)
  ))
_sym_db.RegisterMessage(DetectionLossParameter)

RegionLossParameter = _reflection.GeneratedProtocolMessageType('RegionLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _REGIONLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.RegionLossParameter)
  ))
_sym_db.RegisterMessage(RegionLossParameter)

ReorgParameter = _reflection.GeneratedProtocolMessageType('ReorgParameter', (_message.Message,), dict(
  DESCRIPTOR = _REORGPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ReorgParameter)
  ))
_sym_db.RegisterMessage(ReorgParameter)

EvalDetectionParameter = _reflection.GeneratedProtocolMessageType('EvalDetectionParameter', (_message.Message,), dict(
  DESCRIPTOR = _EVALDETECTIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.EvalDetectionParameter)
  ))
_sym_db.RegisterMessage(EvalDetectionParameter)

ConvolutionParameter = _reflection.GeneratedProtocolMessageType('ConvolutionParameter', (_message.Message,), dict(
  DESCRIPTOR = _CONVOLUTIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ConvolutionParameter)
  ))
_sym_db.RegisterMessage(ConvolutionParameter)

CropParameter = _reflection.GeneratedProtocolMessageType('CropParameter', (_message.Message,), dict(
  DESCRIPTOR = _CROPPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.CropParameter)
  ))
_sym_db.RegisterMessage(CropParameter)

DataParameter = _reflection.GeneratedProtocolMessageType('DataParameter', (_message.Message,), dict(
  DESCRIPTOR = _DATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DataParameter)
  ))
_sym_db.RegisterMessage(DataParameter)

DetectionEvaluateParameter = _reflection.GeneratedProtocolMessageType('DetectionEvaluateParameter', (_message.Message,), dict(
  DESCRIPTOR = _DETECTIONEVALUATEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DetectionEvaluateParameter)
  ))
_sym_db.RegisterMessage(DetectionEvaluateParameter)

NonMaximumSuppressionParameter = _reflection.GeneratedProtocolMessageType('NonMaximumSuppressionParameter', (_message.Message,), dict(
  DESCRIPTOR = _NONMAXIMUMSUPPRESSIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.NonMaximumSuppressionParameter)
  ))
_sym_db.RegisterMessage(NonMaximumSuppressionParameter)

SaveOutputParameter = _reflection.GeneratedProtocolMessageType('SaveOutputParameter', (_message.Message,), dict(
  DESCRIPTOR = _SAVEOUTPUTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SaveOutputParameter)
  ))
_sym_db.RegisterMessage(SaveOutputParameter)

DetectionOutputParameter = _reflection.GeneratedProtocolMessageType('DetectionOutputParameter', (_message.Message,), dict(
  DESCRIPTOR = _DETECTIONOUTPUTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DetectionOutputParameter)
  ))
_sym_db.RegisterMessage(DetectionOutputParameter)

DropoutParameter = _reflection.GeneratedProtocolMessageType('DropoutParameter', (_message.Message,), dict(
  DESCRIPTOR = _DROPOUTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DropoutParameter)
  ))
_sym_db.RegisterMessage(DropoutParameter)

DummyDataParameter = _reflection.GeneratedProtocolMessageType('DummyDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _DUMMYDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DummyDataParameter)
  ))
_sym_db.RegisterMessage(DummyDataParameter)

EltwiseParameter = _reflection.GeneratedProtocolMessageType('EltwiseParameter', (_message.Message,), dict(
  DESCRIPTOR = _ELTWISEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.EltwiseParameter)
  ))
_sym_db.RegisterMessage(EltwiseParameter)

ELUParameter = _reflection.GeneratedProtocolMessageType('ELUParameter', (_message.Message,), dict(
  DESCRIPTOR = _ELUPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ELUParameter)
  ))
_sym_db.RegisterMessage(ELUParameter)

EmbedParameter = _reflection.GeneratedProtocolMessageType('EmbedParameter', (_message.Message,), dict(
  DESCRIPTOR = _EMBEDPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.EmbedParameter)
  ))
_sym_db.RegisterMessage(EmbedParameter)

ExpParameter = _reflection.GeneratedProtocolMessageType('ExpParameter', (_message.Message,), dict(
  DESCRIPTOR = _EXPPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ExpParameter)
  ))
_sym_db.RegisterMessage(ExpParameter)

FlattenParameter = _reflection.GeneratedProtocolMessageType('FlattenParameter', (_message.Message,), dict(
  DESCRIPTOR = _FLATTENPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.FlattenParameter)
  ))
_sym_db.RegisterMessage(FlattenParameter)

HDF5DataParameter = _reflection.GeneratedProtocolMessageType('HDF5DataParameter', (_message.Message,), dict(
  DESCRIPTOR = _HDF5DATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.HDF5DataParameter)
  ))
_sym_db.RegisterMessage(HDF5DataParameter)

HDF5OutputParameter = _reflection.GeneratedProtocolMessageType('HDF5OutputParameter', (_message.Message,), dict(
  DESCRIPTOR = _HDF5OUTPUTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.HDF5OutputParameter)
  ))
_sym_db.RegisterMessage(HDF5OutputParameter)

HingeLossParameter = _reflection.GeneratedProtocolMessageType('HingeLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _HINGELOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.HingeLossParameter)
  ))
_sym_db.RegisterMessage(HingeLossParameter)

ImageDataParameter = _reflection.GeneratedProtocolMessageType('ImageDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ImageDataParameter)
  ))
_sym_db.RegisterMessage(ImageDataParameter)

InfogainLossParameter = _reflection.GeneratedProtocolMessageType('InfogainLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _INFOGAINLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.InfogainLossParameter)
  ))
_sym_db.RegisterMessage(InfogainLossParameter)

InnerProductParameter = _reflection.GeneratedProtocolMessageType('InnerProductParameter', (_message.Message,), dict(
  DESCRIPTOR = _INNERPRODUCTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.InnerProductParameter)
  ))
_sym_db.RegisterMessage(InnerProductParameter)

InputParameter = _reflection.GeneratedProtocolMessageType('InputParameter', (_message.Message,), dict(
  DESCRIPTOR = _INPUTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.InputParameter)
  ))
_sym_db.RegisterMessage(InputParameter)

LogParameter = _reflection.GeneratedProtocolMessageType('LogParameter', (_message.Message,), dict(
  DESCRIPTOR = _LOGPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.LogParameter)
  ))
_sym_db.RegisterMessage(LogParameter)

LRNParameter = _reflection.GeneratedProtocolMessageType('LRNParameter', (_message.Message,), dict(
  DESCRIPTOR = _LRNPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.LRNParameter)
  ))
_sym_db.RegisterMessage(LRNParameter)

MemoryDataParameter = _reflection.GeneratedProtocolMessageType('MemoryDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _MEMORYDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MemoryDataParameter)
  ))
_sym_db.RegisterMessage(MemoryDataParameter)

MultiBoxLossParameter = _reflection.GeneratedProtocolMessageType('MultiBoxLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _MULTIBOXLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MultiBoxLossParameter)
  ))
_sym_db.RegisterMessage(MultiBoxLossParameter)

PermuteParameter = _reflection.GeneratedProtocolMessageType('PermuteParameter', (_message.Message,), dict(
  DESCRIPTOR = _PERMUTEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PermuteParameter)
  ))
_sym_db.RegisterMessage(PermuteParameter)

MVNParameter = _reflection.GeneratedProtocolMessageType('MVNParameter', (_message.Message,), dict(
  DESCRIPTOR = _MVNPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MVNParameter)
  ))
_sym_db.RegisterMessage(MVNParameter)

ParameterParameter = _reflection.GeneratedProtocolMessageType('ParameterParameter', (_message.Message,), dict(
  DESCRIPTOR = _PARAMETERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ParameterParameter)
  ))
_sym_db.RegisterMessage(ParameterParameter)

PoolingParameter = _reflection.GeneratedProtocolMessageType('PoolingParameter', (_message.Message,), dict(
  DESCRIPTOR = _POOLINGPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PoolingParameter)
  ))
_sym_db.RegisterMessage(PoolingParameter)

PowerParameter = _reflection.GeneratedProtocolMessageType('PowerParameter', (_message.Message,), dict(
  DESCRIPTOR = _POWERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PowerParameter)
  ))
_sym_db.RegisterMessage(PowerParameter)

PriorBoxParameter = _reflection.GeneratedProtocolMessageType('PriorBoxParameter', (_message.Message,), dict(
  DESCRIPTOR = _PRIORBOXPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PriorBoxParameter)
  ))
_sym_db.RegisterMessage(PriorBoxParameter)

PythonParameter = _reflection.GeneratedProtocolMessageType('PythonParameter', (_message.Message,), dict(
  DESCRIPTOR = _PYTHONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PythonParameter)
  ))
_sym_db.RegisterMessage(PythonParameter)

RecurrentParameter = _reflection.GeneratedProtocolMessageType('RecurrentParameter', (_message.Message,), dict(
  DESCRIPTOR = _RECURRENTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.RecurrentParameter)
  ))
_sym_db.RegisterMessage(RecurrentParameter)

ReductionParameter = _reflection.GeneratedProtocolMessageType('ReductionParameter', (_message.Message,), dict(
  DESCRIPTOR = _REDUCTIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ReductionParameter)
  ))
_sym_db.RegisterMessage(ReductionParameter)

ReLUParameter = _reflection.GeneratedProtocolMessageType('ReLUParameter', (_message.Message,), dict(
  DESCRIPTOR = _RELUPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ReLUParameter)
  ))
_sym_db.RegisterMessage(ReLUParameter)

ReshapeParameter = _reflection.GeneratedProtocolMessageType('ReshapeParameter', (_message.Message,), dict(
  DESCRIPTOR = _RESHAPEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ReshapeParameter)
  ))
_sym_db.RegisterMessage(ReshapeParameter)

ROIPoolingParameter = _reflection.GeneratedProtocolMessageType('ROIPoolingParameter', (_message.Message,), dict(
  DESCRIPTOR = _ROIPOOLINGPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ROIPoolingParameter)
  ))
_sym_db.RegisterMessage(ROIPoolingParameter)

ScaleParameter = _reflection.GeneratedProtocolMessageType('ScaleParameter', (_message.Message,), dict(
  DESCRIPTOR = _SCALEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ScaleParameter)
  ))
_sym_db.RegisterMessage(ScaleParameter)

SigmoidParameter = _reflection.GeneratedProtocolMessageType('SigmoidParameter', (_message.Message,), dict(
  DESCRIPTOR = _SIGMOIDPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SigmoidParameter)
  ))
_sym_db.RegisterMessage(SigmoidParameter)

SmoothL1LossParameter = _reflection.GeneratedProtocolMessageType('SmoothL1LossParameter', (_message.Message,), dict(
  DESCRIPTOR = _SMOOTHL1LOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SmoothL1LossParameter)
  ))
_sym_db.RegisterMessage(SmoothL1LossParameter)

SliceParameter = _reflection.GeneratedProtocolMessageType('SliceParameter', (_message.Message,), dict(
  DESCRIPTOR = _SLICEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SliceParameter)
  ))
_sym_db.RegisterMessage(SliceParameter)

SoftmaxParameter = _reflection.GeneratedProtocolMessageType('SoftmaxParameter', (_message.Message,), dict(
  DESCRIPTOR = _SOFTMAXPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SoftmaxParameter)
  ))
_sym_db.RegisterMessage(SoftmaxParameter)

TanHParameter = _reflection.GeneratedProtocolMessageType('TanHParameter', (_message.Message,), dict(
  DESCRIPTOR = _TANHPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.TanHParameter)
  ))
_sym_db.RegisterMessage(TanHParameter)

TileParameter = _reflection.GeneratedProtocolMessageType('TileParameter', (_message.Message,), dict(
  DESCRIPTOR = _TILEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.TileParameter)
  ))
_sym_db.RegisterMessage(TileParameter)

ThresholdParameter = _reflection.GeneratedProtocolMessageType('ThresholdParameter', (_message.Message,), dict(
  DESCRIPTOR = _THRESHOLDPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ThresholdParameter)
  ))
_sym_db.RegisterMessage(ThresholdParameter)

WindowDataParameter = _reflection.GeneratedProtocolMessageType('WindowDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _WINDOWDATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.WindowDataParameter)
  ))
_sym_db.RegisterMessage(WindowDataParameter)

SPPParameter = _reflection.GeneratedProtocolMessageType('SPPParameter', (_message.Message,), dict(
  DESCRIPTOR = _SPPPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.SPPParameter)
  ))
_sym_db.RegisterMessage(SPPParameter)

V1LayerParameter = _reflection.GeneratedProtocolMessageType('V1LayerParameter', (_message.Message,), dict(
  DESCRIPTOR = _V1LAYERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.V1LayerParameter)
  ))
_sym_db.RegisterMessage(V1LayerParameter)

V0LayerParameter = _reflection.GeneratedProtocolMessageType('V0LayerParameter', (_message.Message,), dict(
  DESCRIPTOR = _V0LAYERPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.V0LayerParameter)
  ))
_sym_db.RegisterMessage(V0LayerParameter)

PReLUParameter = _reflection.GeneratedProtocolMessageType('PReLUParameter', (_message.Message,), dict(
  DESCRIPTOR = _PRELUPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.PReLUParameter)
  ))
_sym_db.RegisterMessage(PReLUParameter)

RPNParameter = _reflection.GeneratedProtocolMessageType('RPNParameter', (_message.Message,), dict(
  DESCRIPTOR = _RPNPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.RPNParameter)
  ))
_sym_db.RegisterMessage(RPNParameter)

VideoDataParameter = _reflection.GeneratedProtocolMessageType('VideoDataParameter', (_message.Message,), dict(
  DESCRIPTOR = _VIDEODATAPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.VideoDataParameter)
  ))
_sym_db.RegisterMessage(VideoDataParameter)

CenterLossParameter = _reflection.GeneratedProtocolMessageType('CenterLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _CENTERLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.CenterLossParameter)
  ))
_sym_db.RegisterMessage(CenterLossParameter)

MarginInnerProductParameter = _reflection.GeneratedProtocolMessageType('MarginInnerProductParameter', (_message.Message,), dict(
  DESCRIPTOR = _MARGININNERPRODUCTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.MarginInnerProductParameter)
  ))
_sym_db.RegisterMessage(MarginInnerProductParameter)

AdditiveMarginInnerProductParameter = _reflection.GeneratedProtocolMessageType('AdditiveMarginInnerProductParameter', (_message.Message,), dict(
  DESCRIPTOR = _ADDITIVEMARGININNERPRODUCTPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.AdditiveMarginInnerProductParameter)
  ))
_sym_db.RegisterMessage(AdditiveMarginInnerProductParameter)

DeformableConvolutionParameter = _reflection.GeneratedProtocolMessageType('DeformableConvolutionParameter', (_message.Message,), dict(
  DESCRIPTOR = _DEFORMABLECONVOLUTIONPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.DeformableConvolutionParameter)
  ))
_sym_db.RegisterMessage(DeformableConvolutionParameter)

LabelSpecificAddParameter = _reflection.GeneratedProtocolMessageType('LabelSpecificAddParameter', (_message.Message,), dict(
  DESCRIPTOR = _LABELSPECIFICADDPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.LabelSpecificAddParameter)
  ))
_sym_db.RegisterMessage(LabelSpecificAddParameter)

ChannelScaleParameter = _reflection.GeneratedProtocolMessageType('ChannelScaleParameter', (_message.Message,), dict(
  DESCRIPTOR = _CHANNELSCALEPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ChannelScaleParameter)
  ))
_sym_db.RegisterMessage(ChannelScaleParameter)

CosinAddmParameter = _reflection.GeneratedProtocolMessageType('CosinAddmParameter', (_message.Message,), dict(
  DESCRIPTOR = _COSINADDMPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.CosinAddmParameter)
  ))
_sym_db.RegisterMessage(CosinAddmParameter)

CosinMulmParameter = _reflection.GeneratedProtocolMessageType('CosinMulmParameter', (_message.Message,), dict(
  DESCRIPTOR = _COSINMULMPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.CosinMulmParameter)
  ))
_sym_db.RegisterMessage(CosinMulmParameter)

CoupledClusterLossParameter = _reflection.GeneratedProtocolMessageType('CoupledClusterLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _COUPLEDCLUSTERLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.CoupledClusterLossParameter)
  ))
_sym_db.RegisterMessage(CoupledClusterLossParameter)

TripletLossParameter = _reflection.GeneratedProtocolMessageType('TripletLossParameter', (_message.Message,), dict(
  DESCRIPTOR = _TRIPLETLOSSPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.TripletLossParameter)
  ))
_sym_db.RegisterMessage(TripletLossParameter)

GeneralTripletParameter = _reflection.GeneratedProtocolMessageType('GeneralTripletParameter', (_message.Message,), dict(
  DESCRIPTOR = _GENERALTRIPLETPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.GeneralTripletParameter)
  ))
_sym_db.RegisterMessage(GeneralTripletParameter)

ROIAlignParameter = _reflection.GeneratedProtocolMessageType('ROIAlignParameter', (_message.Message,), dict(
  DESCRIPTOR = _ROIALIGNPARAMETER,
  __module__ = 'caffe_pb2'
  # @@protoc_insertion_point(class_scope:caffe.ROIAlignParameter)
  ))
_sym_db.RegisterMessage(ROIAlignParameter)


_BLOBSHAPE.fields_by_name['dim']._options = None
_BLOBPROTO.fields_by_name['data']._options = None
_BLOBPROTO.fields_by_name['diff']._options = None
_BLOBPROTO.fields_by_name['double_data']._options = None
_BLOBPROTO.fields_by_name['double_diff']._options = None
# @@protoc_insertion_point(module_scope)
