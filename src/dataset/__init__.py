# Dataset module - HRTF datasets
from .hrtf import SonicomDataSet, OnlyHRTFDataSet, SingleSubjectDataSet

# 兼容别名
WidedspreadDataSet = SonicomDataSet
WidedspreadOnlyHRTFDataSet = OnlyHRTFDataSet
