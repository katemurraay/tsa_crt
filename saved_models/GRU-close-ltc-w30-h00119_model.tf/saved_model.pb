Ƙ
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.4.12unknown8��
|
dense_156/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Kd*!
shared_namedense_156/kernel
u
$dense_156/kernel/Read/ReadVariableOpReadVariableOpdense_156/kernel*
_output_shapes

:Kd*
dtype0
t
dense_156/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_156/bias
m
"dense_156/bias/Read/ReadVariableOpReadVariableOpdense_156/bias*
_output_shapes
:d*
dtype0
|
dense_157/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_157/kernel
u
$dense_157/kernel/Read/ReadVariableOpReadVariableOpdense_157/kernel*
_output_shapes

:d*
dtype0
t
dense_157/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_157/bias
m
"dense_157/bias/Read/ReadVariableOpReadVariableOpdense_157/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
gru_28/gru_cell_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_namegru_28/gru_cell_28/kernel
�
-gru_28/gru_cell_28/kernel/Read/ReadVariableOpReadVariableOpgru_28/gru_cell_28/kernel*
_output_shapes
:	�*
dtype0
�
#gru_28/gru_cell_28/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K�*4
shared_name%#gru_28/gru_cell_28/recurrent_kernel
�
7gru_28/gru_cell_28/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_28/gru_cell_28/recurrent_kernel*
_output_shapes
:	K�*
dtype0
�
gru_28/gru_cell_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namegru_28/gru_cell_28/bias
�
+gru_28/gru_cell_28/bias/Read/ReadVariableOpReadVariableOpgru_28/gru_cell_28/bias*
_output_shapes
:	�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/dense_156/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Kd*(
shared_nameAdam/dense_156/kernel/m
�
+Adam/dense_156/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_156/kernel/m*
_output_shapes

:Kd*
dtype0
�
Adam/dense_156/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_156/bias/m
{
)Adam/dense_156/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_156/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_157/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_157/kernel/m
�
+Adam/dense_157/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_157/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/dense_157/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_157/bias/m
{
)Adam/dense_157/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_157/bias/m*
_output_shapes
:*
dtype0
�
 Adam/gru_28/gru_cell_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" Adam/gru_28/gru_cell_28/kernel/m
�
4Adam/gru_28/gru_cell_28/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_28/gru_cell_28/kernel/m*
_output_shapes
:	�*
dtype0
�
*Adam/gru_28/gru_cell_28/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K�*;
shared_name,*Adam/gru_28/gru_cell_28/recurrent_kernel/m
�
>Adam/gru_28/gru_cell_28/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_28/gru_cell_28/recurrent_kernel/m*
_output_shapes
:	K�*
dtype0
�
Adam/gru_28/gru_cell_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_28/gru_cell_28/bias/m
�
2Adam/gru_28/gru_cell_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_28/gru_cell_28/bias/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_156/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Kd*(
shared_nameAdam/dense_156/kernel/v
�
+Adam/dense_156/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_156/kernel/v*
_output_shapes

:Kd*
dtype0
�
Adam/dense_156/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_156/bias/v
{
)Adam/dense_156/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_156/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_157/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_157/kernel/v
�
+Adam/dense_157/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_157/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/dense_157/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_157/bias/v
{
)Adam/dense_157/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_157/bias/v*
_output_shapes
:*
dtype0
�
 Adam/gru_28/gru_cell_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" Adam/gru_28/gru_cell_28/kernel/v
�
4Adam/gru_28/gru_cell_28/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_28/gru_cell_28/kernel/v*
_output_shapes
:	�*
dtype0
�
*Adam/gru_28/gru_cell_28/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K�*;
shared_name,*Adam/gru_28/gru_cell_28/recurrent_kernel/v
�
>Adam/gru_28/gru_cell_28/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_28/gru_cell_28/recurrent_kernel/v*
_output_shapes
:	K�*
dtype0
�
Adam/gru_28/gru_cell_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_28/gru_cell_28/bias/v
�
2Adam/gru_28/gru_cell_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_28/gru_cell_28/bias/v*
_output_shapes
:	�*
dtype0

NoOpNoOp
�.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�-
value�-B�- B�-
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
l

cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�
iter

beta_1

beta_2
	decay
 learning_ratemSmTmUmV!mW"mX#mYvZv[v\v]!v^"v_#v`
1
!0
"1
#2
3
4
5
6
1
!0
"1
#2
3
4
5
6
 
�
trainable_variables
$layer_metrics
	variables
%metrics

&layers
'layer_regularization_losses
regularization_losses
(non_trainable_variables
 
~

!kernel
"recurrent_kernel
#bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
 

!0
"1
#2

!0
"1
#2
 
�
trainable_variables
-layer_metrics
	variables

.states
/metrics

0layers
1layer_regularization_losses
regularization_losses
2non_trainable_variables
\Z
VARIABLE_VALUEdense_156/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_156/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
3layer_metrics
	variables
4metrics

5layers
6layer_regularization_losses
regularization_losses
7non_trainable_variables
\Z
VARIABLE_VALUEdense_157/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_157/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
8layer_metrics
	variables
9metrics

:layers
;layer_regularization_losses
regularization_losses
<non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_28/gru_cell_28/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_28/gru_cell_28/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_28/gru_cell_28/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1
?2

0
1
2
 
 

!0
"1
#2

!0
"1
#2
 
�
)trainable_variables
@layer_metrics
*	variables
Ametrics

Blayers
Clayer_regularization_losses
+regularization_losses
Dnon_trainable_variables
 
 
 


0
 
 
 
 
 
 
 
 
 
 
 
 
4
	Etotal
	Fcount
G	variables
H	keras_api
D
	Itotal
	Jcount
K
_fn_kwargs
L	variables
M	keras_api
D
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

G	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

L	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

Q	variables
}
VARIABLE_VALUEAdam/dense_156/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_156/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_157/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_157/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/gru_28/gru_cell_28/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/gru_28/gru_cell_28/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/gru_28/gru_cell_28/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_156/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_156/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_157/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_157/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/gru_28/gru_cell_28/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/gru_28/gru_cell_28/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/gru_28/gru_cell_28/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_gru_28_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_28_inputgru_28/gru_cell_28/biasgru_28/gru_cell_28/kernel#gru_28/gru_cell_28/recurrent_kerneldense_156/kerneldense_156/biasdense_157/kerneldense_157/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_32655690
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_156/kernel/Read/ReadVariableOp"dense_156/bias/Read/ReadVariableOp$dense_157/kernel/Read/ReadVariableOp"dense_157/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-gru_28/gru_cell_28/kernel/Read/ReadVariableOp7gru_28/gru_cell_28/recurrent_kernel/Read/ReadVariableOp+gru_28/gru_cell_28/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/dense_156/kernel/m/Read/ReadVariableOp)Adam/dense_156/bias/m/Read/ReadVariableOp+Adam/dense_157/kernel/m/Read/ReadVariableOp)Adam/dense_157/bias/m/Read/ReadVariableOp4Adam/gru_28/gru_cell_28/kernel/m/Read/ReadVariableOp>Adam/gru_28/gru_cell_28/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_28/gru_cell_28/bias/m/Read/ReadVariableOp+Adam/dense_156/kernel/v/Read/ReadVariableOp)Adam/dense_156/bias/v/Read/ReadVariableOp+Adam/dense_157/kernel/v/Read/ReadVariableOp)Adam/dense_157/bias/v/Read/ReadVariableOp4Adam/gru_28/gru_cell_28/kernel/v/Read/ReadVariableOp>Adam/gru_28/gru_cell_28/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_28/gru_cell_28/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_32657018
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_156/kerneldense_156/biasdense_157/kerneldense_157/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_28/gru_cell_28/kernel#gru_28/gru_cell_28/recurrent_kernelgru_28/gru_cell_28/biastotalcounttotal_1count_1total_2count_2Adam/dense_156/kernel/mAdam/dense_156/bias/mAdam/dense_157/kernel/mAdam/dense_157/bias/m Adam/gru_28/gru_cell_28/kernel/m*Adam/gru_28/gru_cell_28/recurrent_kernel/mAdam/gru_28/gru_cell_28/bias/mAdam/dense_156/kernel/vAdam/dense_156/bias/vAdam/dense_157/kernel/vAdam/dense_157/bias/v Adam/gru_28/gru_cell_28/kernel/v*Adam/gru_28/gru_cell_28/recurrent_kernel/vAdam/gru_28/gru_cell_28/bias/v*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_32657124��
�[
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32656231
inputs_0'
#gru_cell_28_readvariableop_resource.
*gru_cell_28_matmul_readvariableop_resource0
,gru_cell_28_matmul_1_readvariableop_resource
identity��!gru_cell_28/MatMul/ReadVariableOp�#gru_cell_28/MatMul_1/ReadVariableOp�gru_cell_28/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_28/ReadVariableOpReadVariableOp#gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_28/ReadVariableOp�
gru_cell_28/unstackUnpack"gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_28/unstack�
!gru_cell_28/MatMul/ReadVariableOpReadVariableOp*gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_cell_28/MatMul/ReadVariableOp�
gru_cell_28/MatMulMatMulstrided_slice_2:output:0)gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul�
gru_cell_28/BiasAddBiasAddgru_cell_28/MatMul:product:0gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAddh
gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_28/Const�
gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split/split_dim�
gru_cell_28/splitSplit$gru_cell_28/split/split_dim:output:0gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split�
#gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02%
#gru_cell_28/MatMul_1/ReadVariableOp�
gru_cell_28/MatMul_1MatMulzeros:output:0+gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul_1�
gru_cell_28/BiasAdd_1BiasAddgru_cell_28/MatMul_1:product:0gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAdd_1
gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_28/Const_1�
gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split_1/split_dim�
gru_cell_28/split_1SplitVgru_cell_28/BiasAdd_1:output:0gru_cell_28/Const_1:output:0&gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split_1�
gru_cell_28/addAddV2gru_cell_28/split:output:0gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add|
gru_cell_28/SigmoidSigmoidgru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid�
gru_cell_28/add_1AddV2gru_cell_28/split:output:1gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_1�
gru_cell_28/Sigmoid_1Sigmoidgru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid_1�
gru_cell_28/mulMulgru_cell_28/Sigmoid_1:y:0gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul�
gru_cell_28/add_2AddV2gru_cell_28/split:output:2gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_2u
gru_cell_28/ReluRelugru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Relu�
gru_cell_28/mul_1Mulgru_cell_28/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_1k
gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_28/sub/x�
gru_cell_28/subSubgru_cell_28/sub/x:output:0gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/sub�
gru_cell_28/mul_2Mulgru_cell_28/sub:z:0gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_2�
gru_cell_28/add_3AddV2gru_cell_28/mul_1:z:0gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_28_readvariableop_resource*gru_cell_28_matmul_readvariableop_resource,gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32656141*
condR
while_cond_32656140*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^gru_cell_28/MatMul/ReadVariableOp$^gru_cell_28/MatMul_1/ReadVariableOp^gru_cell_28/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2F
!gru_cell_28/MatMul/ReadVariableOp!gru_cell_28/MatMul/ReadVariableOp2J
#gru_cell_28/MatMul_1/ReadVariableOp#gru_cell_28/MatMul_1/ReadVariableOp28
gru_cell_28/ReadVariableOpgru_cell_28/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�[
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32656730

inputs'
#gru_cell_28_readvariableop_resource.
*gru_cell_28_matmul_readvariableop_resource0
,gru_cell_28_matmul_1_readvariableop_resource
identity��!gru_cell_28/MatMul/ReadVariableOp�#gru_cell_28/MatMul_1/ReadVariableOp�gru_cell_28/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_28/ReadVariableOpReadVariableOp#gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_28/ReadVariableOp�
gru_cell_28/unstackUnpack"gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_28/unstack�
!gru_cell_28/MatMul/ReadVariableOpReadVariableOp*gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_cell_28/MatMul/ReadVariableOp�
gru_cell_28/MatMulMatMulstrided_slice_2:output:0)gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul�
gru_cell_28/BiasAddBiasAddgru_cell_28/MatMul:product:0gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAddh
gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_28/Const�
gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split/split_dim�
gru_cell_28/splitSplit$gru_cell_28/split/split_dim:output:0gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split�
#gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02%
#gru_cell_28/MatMul_1/ReadVariableOp�
gru_cell_28/MatMul_1MatMulzeros:output:0+gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul_1�
gru_cell_28/BiasAdd_1BiasAddgru_cell_28/MatMul_1:product:0gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAdd_1
gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_28/Const_1�
gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split_1/split_dim�
gru_cell_28/split_1SplitVgru_cell_28/BiasAdd_1:output:0gru_cell_28/Const_1:output:0&gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split_1�
gru_cell_28/addAddV2gru_cell_28/split:output:0gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add|
gru_cell_28/SigmoidSigmoidgru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid�
gru_cell_28/add_1AddV2gru_cell_28/split:output:1gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_1�
gru_cell_28/Sigmoid_1Sigmoidgru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid_1�
gru_cell_28/mulMulgru_cell_28/Sigmoid_1:y:0gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul�
gru_cell_28/add_2AddV2gru_cell_28/split:output:2gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_2u
gru_cell_28/ReluRelugru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Relu�
gru_cell_28/mul_1Mulgru_cell_28/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_1k
gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_28/sub/x�
gru_cell_28/subSubgru_cell_28/sub/x:output:0gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/sub�
gru_cell_28/mul_2Mulgru_cell_28/sub:z:0gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_2�
gru_cell_28/add_3AddV2gru_cell_28/mul_1:z:0gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_28_readvariableop_resource*gru_cell_28_matmul_readvariableop_resource,gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32656640*
condR
while_cond_32656639*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^gru_cell_28/MatMul/ReadVariableOp$^gru_cell_28/MatMul_1/ReadVariableOp^gru_cell_28/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2F
!gru_cell_28/MatMul/ReadVariableOp!gru_cell_28/MatMul/ReadVariableOp2J
#gru_cell_28/MatMul_1/ReadVariableOp#gru_cell_28/MatMul_1/ReadVariableOp28
gru_cell_28/ReadVariableOpgru_cell_28/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655580
gru_28_input
gru_28_32655562
gru_28_32655564
gru_28_32655566
dense_156_32655569
dense_156_32655571
dense_157_32655574
dense_157_32655576
identity��!dense_156/StatefulPartitionedCall�!dense_157/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallgru_28_inputgru_28_32655562gru_28_32655564gru_28_32655566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326554752 
gru_28/StatefulPartitionedCall�
!dense_156/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0dense_156_32655569dense_156_32655571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_326555162#
!dense_156/StatefulPartitionedCall�
!dense_157/StatefulPartitionedCallStatefulPartitionedCall*dense_156/StatefulPartitionedCall:output:0dense_157_32655574dense_157_32655576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_326555422#
!dense_157/StatefulPartitionedCall�
IdentityIdentity*dense_157/StatefulPartitionedCall:output:0"^dense_156/StatefulPartitionedCall"^dense_157/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2F
!dense_156/StatefulPartitionedCall!dense_156/StatefulPartitionedCall2F
!dense_157/StatefulPartitionedCall!dense_157/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_namegru_28_input
�	
�
.__inference_gru_cell_28_layer_call_fn_32656885

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_326546642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�
�
while_cond_32656480
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32656480___redundant_placeholder06
2while_while_cond_32656480___redundant_placeholder16
2while_while_cond_32656480___redundant_placeholder26
2while_while_cond_32656480___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_32656140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32656140___redundant_placeholder06
2while_while_cond_32656140___redundant_placeholder16
2while_while_cond_32656140___redundant_placeholder26
2while_while_cond_32656140___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�[
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32656571

inputs'
#gru_cell_28_readvariableop_resource.
*gru_cell_28_matmul_readvariableop_resource0
,gru_cell_28_matmul_1_readvariableop_resource
identity��!gru_cell_28/MatMul/ReadVariableOp�#gru_cell_28/MatMul_1/ReadVariableOp�gru_cell_28/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_28/ReadVariableOpReadVariableOp#gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_28/ReadVariableOp�
gru_cell_28/unstackUnpack"gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_28/unstack�
!gru_cell_28/MatMul/ReadVariableOpReadVariableOp*gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_cell_28/MatMul/ReadVariableOp�
gru_cell_28/MatMulMatMulstrided_slice_2:output:0)gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul�
gru_cell_28/BiasAddBiasAddgru_cell_28/MatMul:product:0gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAddh
gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_28/Const�
gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split/split_dim�
gru_cell_28/splitSplit$gru_cell_28/split/split_dim:output:0gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split�
#gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02%
#gru_cell_28/MatMul_1/ReadVariableOp�
gru_cell_28/MatMul_1MatMulzeros:output:0+gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul_1�
gru_cell_28/BiasAdd_1BiasAddgru_cell_28/MatMul_1:product:0gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAdd_1
gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_28/Const_1�
gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split_1/split_dim�
gru_cell_28/split_1SplitVgru_cell_28/BiasAdd_1:output:0gru_cell_28/Const_1:output:0&gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split_1�
gru_cell_28/addAddV2gru_cell_28/split:output:0gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add|
gru_cell_28/SigmoidSigmoidgru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid�
gru_cell_28/add_1AddV2gru_cell_28/split:output:1gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_1�
gru_cell_28/Sigmoid_1Sigmoidgru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid_1�
gru_cell_28/mulMulgru_cell_28/Sigmoid_1:y:0gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul�
gru_cell_28/add_2AddV2gru_cell_28/split:output:2gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_2u
gru_cell_28/ReluRelugru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Relu�
gru_cell_28/mul_1Mulgru_cell_28/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_1k
gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_28/sub/x�
gru_cell_28/subSubgru_cell_28/sub/x:output:0gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/sub�
gru_cell_28/mul_2Mulgru_cell_28/sub:z:0gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_2�
gru_cell_28/add_3AddV2gru_cell_28/mul_1:z:0gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_28_readvariableop_resource*gru_cell_28_matmul_readvariableop_resource,gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32656481*
condR
while_cond_32656480*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^gru_cell_28/MatMul/ReadVariableOp$^gru_cell_28/MatMul_1/ReadVariableOp^gru_cell_28/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2F
!gru_cell_28/MatMul/ReadVariableOp!gru_cell_28/MatMul/ReadVariableOp2J
#gru_cell_28/MatMul_1/ReadVariableOp#gru_cell_28/MatMul_1/ReadVariableOp28
gru_cell_28/ReadVariableOpgru_cell_28/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_156_layer_call_fn_32656772

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_326555162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������K::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������K
 
_user_specified_nameinputs
�
�
while_cond_32656299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32656299___redundant_placeholder06
2while_while_cond_32656299___redundant_placeholder16
2while_while_cond_32656299___redundant_placeholder26
2while_while_cond_32656299___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�G
�
!__inference__traced_save_32657018
file_prefix/
+savev2_dense_156_kernel_read_readvariableop-
)savev2_dense_156_bias_read_readvariableop/
+savev2_dense_157_kernel_read_readvariableop-
)savev2_dense_157_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_gru_28_gru_cell_28_kernel_read_readvariableopB
>savev2_gru_28_gru_cell_28_recurrent_kernel_read_readvariableop6
2savev2_gru_28_gru_cell_28_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_dense_156_kernel_m_read_readvariableop4
0savev2_adam_dense_156_bias_m_read_readvariableop6
2savev2_adam_dense_157_kernel_m_read_readvariableop4
0savev2_adam_dense_157_bias_m_read_readvariableop?
;savev2_adam_gru_28_gru_cell_28_kernel_m_read_readvariableopI
Esavev2_adam_gru_28_gru_cell_28_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_28_gru_cell_28_bias_m_read_readvariableop6
2savev2_adam_dense_156_kernel_v_read_readvariableop4
0savev2_adam_dense_156_bias_v_read_readvariableop6
2savev2_adam_dense_157_kernel_v_read_readvariableop4
0savev2_adam_dense_157_bias_v_read_readvariableop?
;savev2_adam_gru_28_gru_cell_28_kernel_v_read_readvariableopI
Esavev2_adam_gru_28_gru_cell_28_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_28_gru_cell_28_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_156_kernel_read_readvariableop)savev2_dense_156_bias_read_readvariableop+savev2_dense_157_kernel_read_readvariableop)savev2_dense_157_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_gru_28_gru_cell_28_kernel_read_readvariableop>savev2_gru_28_gru_cell_28_recurrent_kernel_read_readvariableop2savev2_gru_28_gru_cell_28_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_dense_156_kernel_m_read_readvariableop0savev2_adam_dense_156_bias_m_read_readvariableop2savev2_adam_dense_157_kernel_m_read_readvariableop0savev2_adam_dense_157_bias_m_read_readvariableop;savev2_adam_gru_28_gru_cell_28_kernel_m_read_readvariableopEsavev2_adam_gru_28_gru_cell_28_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_28_gru_cell_28_bias_m_read_readvariableop2savev2_adam_dense_156_kernel_v_read_readvariableop0savev2_adam_dense_156_bias_v_read_readvariableop2savev2_adam_dense_157_kernel_v_read_readvariableop0savev2_adam_dense_157_bias_v_read_readvariableop;savev2_adam_gru_28_gru_cell_28_kernel_v_read_readvariableopEsavev2_adam_gru_28_gru_cell_28_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_28_gru_cell_28_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :Kd:d:d:: : : : : :	�:	K�:	�: : : : : : :Kd:d:d::	�:	K�:	�:Kd:d:d::	�:	K�:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Kd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	�:%!

_output_shapes
:	K�:%!

_output_shapes
:	�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Kd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	K�:%!

_output_shapes
:	�:$ 

_output_shapes

:Kd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	K�:% !

_output_shapes
:	�:!

_output_shapes
: 
�
�
while_cond_32655384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32655384___redundant_placeholder06
2while_while_cond_32655384___redundant_placeholder16
2while_while_cond_32655384___redundant_placeholder26
2while_while_cond_32655384___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
)__inference_gru_28_layer_call_fn_32656741

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326553162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_157_layer_call_fn_32656791

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_326555422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
0__inference_sequential_78_layer_call_fn_32655661
gru_28_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_326556442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_namegru_28_input
�
�
while_cond_32655080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32655080___redundant_placeholder06
2while_while_cond_32655080___redundant_placeholder16
2while_while_cond_32655080___redundant_placeholder26
2while_while_cond_32655080___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�S
�	
gru_28_while_body_32655931*
&gru_28_while_gru_28_while_loop_counter0
,gru_28_while_gru_28_while_maximum_iterations
gru_28_while_placeholder
gru_28_while_placeholder_1
gru_28_while_placeholder_2)
%gru_28_while_gru_28_strided_slice_1_0e
agru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_06
2gru_28_while_gru_cell_28_readvariableop_resource_0=
9gru_28_while_gru_cell_28_matmul_readvariableop_resource_0?
;gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0
gru_28_while_identity
gru_28_while_identity_1
gru_28_while_identity_2
gru_28_while_identity_3
gru_28_while_identity_4'
#gru_28_while_gru_28_strided_slice_1c
_gru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor4
0gru_28_while_gru_cell_28_readvariableop_resource;
7gru_28_while_gru_cell_28_matmul_readvariableop_resource=
9gru_28_while_gru_cell_28_matmul_1_readvariableop_resource��.gru_28/while/gru_cell_28/MatMul/ReadVariableOp�0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp�'gru_28/while/gru_cell_28/ReadVariableOp�
>gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2@
>gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shape�
0gru_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0gru_28_while_placeholderGgru_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype022
0gru_28/while/TensorArrayV2Read/TensorListGetItem�
'gru_28/while/gru_cell_28/ReadVariableOpReadVariableOp2gru_28_while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'gru_28/while/gru_cell_28/ReadVariableOp�
 gru_28/while/gru_cell_28/unstackUnpack/gru_28/while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2"
 gru_28/while/gru_cell_28/unstack�
.gru_28/while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp9gru_28_while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype020
.gru_28/while/gru_cell_28/MatMul/ReadVariableOp�
gru_28/while/gru_cell_28/MatMulMatMul7gru_28/while/TensorArrayV2Read/TensorListGetItem:item:06gru_28/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
gru_28/while/gru_cell_28/MatMul�
 gru_28/while/gru_cell_28/BiasAddBiasAdd)gru_28/while/gru_cell_28/MatMul:product:0)gru_28/while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2"
 gru_28/while/gru_cell_28/BiasAdd�
gru_28/while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_28/while/gru_cell_28/Const�
(gru_28/while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(gru_28/while/gru_cell_28/split/split_dim�
gru_28/while/gru_cell_28/splitSplit1gru_28/while/gru_cell_28/split/split_dim:output:0)gru_28/while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2 
gru_28/while/gru_cell_28/split�
0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp;gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype022
0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp�
!gru_28/while/gru_cell_28/MatMul_1MatMulgru_28_while_placeholder_28gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!gru_28/while/gru_cell_28/MatMul_1�
"gru_28/while/gru_cell_28/BiasAdd_1BiasAdd+gru_28/while/gru_cell_28/MatMul_1:product:0)gru_28/while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2$
"gru_28/while/gru_cell_28/BiasAdd_1�
 gru_28/while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2"
 gru_28/while/gru_cell_28/Const_1�
*gru_28/while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*gru_28/while/gru_cell_28/split_1/split_dim�
 gru_28/while/gru_cell_28/split_1SplitV+gru_28/while/gru_cell_28/BiasAdd_1:output:0)gru_28/while/gru_cell_28/Const_1:output:03gru_28/while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2"
 gru_28/while/gru_cell_28/split_1�
gru_28/while/gru_cell_28/addAddV2'gru_28/while/gru_cell_28/split:output:0)gru_28/while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/add�
 gru_28/while/gru_cell_28/SigmoidSigmoid gru_28/while/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2"
 gru_28/while/gru_cell_28/Sigmoid�
gru_28/while/gru_cell_28/add_1AddV2'gru_28/while/gru_cell_28/split:output:1)gru_28/while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/add_1�
"gru_28/while/gru_cell_28/Sigmoid_1Sigmoid"gru_28/while/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2$
"gru_28/while/gru_cell_28/Sigmoid_1�
gru_28/while/gru_cell_28/mulMul&gru_28/while/gru_cell_28/Sigmoid_1:y:0)gru_28/while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/mul�
gru_28/while/gru_cell_28/add_2AddV2'gru_28/while/gru_cell_28/split:output:2 gru_28/while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/add_2�
gru_28/while/gru_cell_28/ReluRelu"gru_28/while/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/Relu�
gru_28/while/gru_cell_28/mul_1Mul$gru_28/while/gru_cell_28/Sigmoid:y:0gru_28_while_placeholder_2*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/mul_1�
gru_28/while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
gru_28/while/gru_cell_28/sub/x�
gru_28/while/gru_cell_28/subSub'gru_28/while/gru_cell_28/sub/x:output:0$gru_28/while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/sub�
gru_28/while/gru_cell_28/mul_2Mul gru_28/while/gru_cell_28/sub:z:0+gru_28/while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/mul_2�
gru_28/while/gru_cell_28/add_3AddV2"gru_28/while/gru_cell_28/mul_1:z:0"gru_28/while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/add_3�
1gru_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_28_while_placeholder_1gru_28_while_placeholder"gru_28/while/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_28/while/TensorArrayV2Write/TensorListSetItemj
gru_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/while/add/y�
gru_28/while/addAddV2gru_28_while_placeholdergru_28/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_28/while/addn
gru_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/while/add_1/y�
gru_28/while/add_1AddV2&gru_28_while_gru_28_while_loop_countergru_28/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_28/while/add_1�
gru_28/while/IdentityIdentitygru_28/while/add_1:z:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity�
gru_28/while/Identity_1Identity,gru_28_while_gru_28_while_maximum_iterations/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity_1�
gru_28/while/Identity_2Identitygru_28/while/add:z:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity_2�
gru_28/while/Identity_3IdentityAgru_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity_3�
gru_28/while/Identity_4Identity"gru_28/while/gru_cell_28/add_3:z:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
gru_28/while/Identity_4"L
#gru_28_while_gru_28_strided_slice_1%gru_28_while_gru_28_strided_slice_1_0"x
9gru_28_while_gru_cell_28_matmul_1_readvariableop_resource;gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0"t
7gru_28_while_gru_cell_28_matmul_readvariableop_resource9gru_28_while_gru_cell_28_matmul_readvariableop_resource_0"f
0gru_28_while_gru_cell_28_readvariableop_resource2gru_28_while_gru_cell_28_readvariableop_resource_0"7
gru_28_while_identitygru_28/while/Identity:output:0";
gru_28_while_identity_1 gru_28/while/Identity_1:output:0";
gru_28_while_identity_2 gru_28/while/Identity_2:output:0";
gru_28_while_identity_3 gru_28/while/Identity_3:output:0";
gru_28_while_identity_4 gru_28/while/Identity_4:output:0"�
_gru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensoragru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2`
.gru_28/while/gru_cell_28/MatMul/ReadVariableOp.gru_28/while/gru_cell_28/MatMul/ReadVariableOp2d
0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp2R
'gru_28/while/gru_cell_28/ReadVariableOp'gru_28/while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655644

inputs
gru_28_32655626
gru_28_32655628
gru_28_32655630
dense_156_32655633
dense_156_32655635
dense_157_32655638
dense_157_32655640
identity��!dense_156/StatefulPartitionedCall�!dense_157/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallinputsgru_28_32655626gru_28_32655628gru_28_32655630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326554752 
gru_28/StatefulPartitionedCall�
!dense_156/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0dense_156_32655633dense_156_32655635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_326555162#
!dense_156/StatefulPartitionedCall�
!dense_157/StatefulPartitionedCallStatefulPartitionedCall*dense_156/StatefulPartitionedCall:output:0dense_157_32655638dense_157_32655640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_326555422#
!dense_157/StatefulPartitionedCall�
IdentityIdentity*dense_157/StatefulPartitionedCall:output:0"^dense_156/StatefulPartitionedCall"^dense_157/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2F
!dense_156/StatefulPartitionedCall!dense_156/StatefulPartitionedCall2F
!dense_157/StatefulPartitionedCall!dense_157/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655559
gru_28_input
gru_28_32655498
gru_28_32655500
gru_28_32655502
dense_156_32655527
dense_156_32655529
dense_157_32655553
dense_157_32655555
identity��!dense_156/StatefulPartitionedCall�!dense_157/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallgru_28_inputgru_28_32655498gru_28_32655500gru_28_32655502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326553162 
gru_28/StatefulPartitionedCall�
!dense_156/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0dense_156_32655527dense_156_32655529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_326555162#
!dense_156/StatefulPartitionedCall�
!dense_157/StatefulPartitionedCallStatefulPartitionedCall*dense_156/StatefulPartitionedCall:output:0dense_157_32655553dense_157_32655555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_326555422#
!dense_157/StatefulPartitionedCall�
IdentityIdentity*dense_157/StatefulPartitionedCall:output:0"^dense_156/StatefulPartitionedCall"^dense_157/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2F
!dense_156/StatefulPartitionedCall!dense_156/StatefulPartitionedCall2F
!dense_157/StatefulPartitionedCall!dense_157/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_namegru_28_input
�
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655604

inputs
gru_28_32655586
gru_28_32655588
gru_28_32655590
dense_156_32655593
dense_156_32655595
dense_157_32655598
dense_157_32655600
identity��!dense_156/StatefulPartitionedCall�!dense_157/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallinputsgru_28_32655586gru_28_32655588gru_28_32655590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326553162 
gru_28/StatefulPartitionedCall�
!dense_156/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0dense_156_32655593dense_156_32655595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_326555162#
!dense_156/StatefulPartitionedCall�
!dense_157/StatefulPartitionedCallStatefulPartitionedCall*dense_156/StatefulPartitionedCall:output:0dense_157_32655598dense_157_32655600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_326555422#
!dense_157/StatefulPartitionedCall�
IdentityIdentity*dense_157/StatefulPartitionedCall:output:0"^dense_156/StatefulPartitionedCall"^dense_157/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2F
!dense_156/StatefulPartitionedCall!dense_156/StatefulPartitionedCall2F
!dense_157/StatefulPartitionedCall!dense_157/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_32655690
gru_28_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_326545922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_namegru_28_input
�
�
0__inference_sequential_78_layer_call_fn_32656072

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_326556442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
while_body_32654963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0 
while_gru_cell_28_32654985_0 
while_gru_cell_28_32654987_0 
while_gru_cell_28_32654989_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_28_32654985
while_gru_cell_28_32654987
while_gru_cell_28_32654989��)while/gru_cell_28/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/gru_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_28_32654985_0while_gru_cell_28_32654987_0while_gru_cell_28_32654989_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_326546642+
)while/gru_cell_28/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_28/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/gru_cell_28/StatefulPartitionedCall:output:1*^while/gru_cell_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2
while/Identity_4":
while_gru_cell_28_32654985while_gru_cell_28_32654985_0":
while_gru_cell_28_32654987while_gru_cell_28_32654987_0":
while_gru_cell_28_32654989while_gru_cell_28_32654989_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2V
)while/gru_cell_28/StatefulPartitionedCall)while/gru_cell_28/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
ڈ
�
$__inference__traced_restore_32657124
file_prefix%
!assignvariableop_dense_156_kernel%
!assignvariableop_1_dense_156_bias'
#assignvariableop_2_dense_157_kernel%
!assignvariableop_3_dense_157_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate0
,assignvariableop_9_gru_28_gru_cell_28_kernel;
7assignvariableop_10_gru_28_gru_cell_28_recurrent_kernel/
+assignvariableop_11_gru_28_gru_cell_28_bias
assignvariableop_12_total
assignvariableop_13_count
assignvariableop_14_total_1
assignvariableop_15_count_1
assignvariableop_16_total_2
assignvariableop_17_count_2/
+assignvariableop_18_adam_dense_156_kernel_m-
)assignvariableop_19_adam_dense_156_bias_m/
+assignvariableop_20_adam_dense_157_kernel_m-
)assignvariableop_21_adam_dense_157_bias_m8
4assignvariableop_22_adam_gru_28_gru_cell_28_kernel_mB
>assignvariableop_23_adam_gru_28_gru_cell_28_recurrent_kernel_m6
2assignvariableop_24_adam_gru_28_gru_cell_28_bias_m/
+assignvariableop_25_adam_dense_156_kernel_v-
)assignvariableop_26_adam_dense_156_bias_v/
+assignvariableop_27_adam_dense_157_kernel_v-
)assignvariableop_28_adam_dense_157_bias_v8
4assignvariableop_29_adam_gru_28_gru_cell_28_kernel_vB
>assignvariableop_30_adam_gru_28_gru_cell_28_recurrent_kernel_v6
2assignvariableop_31_adam_gru_28_gru_cell_28_bias_v
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_dense_156_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_156_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_157_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_157_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_gru_28_gru_cell_28_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_gru_28_gru_cell_28_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_gru_28_gru_cell_28_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_dense_156_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_156_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_dense_157_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_157_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_gru_28_gru_cell_28_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_gru_28_gru_cell_28_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_gru_28_gru_cell_28_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_156_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_156_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_157_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_157_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_gru_28_gru_cell_28_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_gru_28_gru_cell_28_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_gru_28_gru_cell_28_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32�
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
)__inference_gru_28_layer_call_fn_32656752

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326554752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_32654962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32654962___redundant_placeholder06
2while_while_cond_32654962___redundant_placeholder16
2while_while_cond_32654962___redundant_placeholder26
2while_while_cond_32654962___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
)__inference_gru_28_layer_call_fn_32656412
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326551452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32656831

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�	
�
G__inference_dense_156_layer_call_and_return_conditional_losses_32656763

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������K::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������K
 
_user_specified_nameinputs
�
�
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32654704

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������K
 
_user_specified_namestates
�[
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32656390
inputs_0'
#gru_cell_28_readvariableop_resource.
*gru_cell_28_matmul_readvariableop_resource0
,gru_cell_28_matmul_1_readvariableop_resource
identity��!gru_cell_28/MatMul/ReadVariableOp�#gru_cell_28/MatMul_1/ReadVariableOp�gru_cell_28/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_28/ReadVariableOpReadVariableOp#gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_28/ReadVariableOp�
gru_cell_28/unstackUnpack"gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_28/unstack�
!gru_cell_28/MatMul/ReadVariableOpReadVariableOp*gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_cell_28/MatMul/ReadVariableOp�
gru_cell_28/MatMulMatMulstrided_slice_2:output:0)gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul�
gru_cell_28/BiasAddBiasAddgru_cell_28/MatMul:product:0gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAddh
gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_28/Const�
gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split/split_dim�
gru_cell_28/splitSplit$gru_cell_28/split/split_dim:output:0gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split�
#gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02%
#gru_cell_28/MatMul_1/ReadVariableOp�
gru_cell_28/MatMul_1MatMulzeros:output:0+gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul_1�
gru_cell_28/BiasAdd_1BiasAddgru_cell_28/MatMul_1:product:0gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAdd_1
gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_28/Const_1�
gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split_1/split_dim�
gru_cell_28/split_1SplitVgru_cell_28/BiasAdd_1:output:0gru_cell_28/Const_1:output:0&gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split_1�
gru_cell_28/addAddV2gru_cell_28/split:output:0gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add|
gru_cell_28/SigmoidSigmoidgru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid�
gru_cell_28/add_1AddV2gru_cell_28/split:output:1gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_1�
gru_cell_28/Sigmoid_1Sigmoidgru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid_1�
gru_cell_28/mulMulgru_cell_28/Sigmoid_1:y:0gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul�
gru_cell_28/add_2AddV2gru_cell_28/split:output:2gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_2u
gru_cell_28/ReluRelugru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Relu�
gru_cell_28/mul_1Mulgru_cell_28/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_1k
gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_28/sub/x�
gru_cell_28/subSubgru_cell_28/sub/x:output:0gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/sub�
gru_cell_28/mul_2Mulgru_cell_28/sub:z:0gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_2�
gru_cell_28/add_3AddV2gru_cell_28/mul_1:z:0gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_28_readvariableop_resource*gru_cell_28_matmul_readvariableop_resource,gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32656300*
condR
while_cond_32656299*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^gru_cell_28/MatMul/ReadVariableOp$^gru_cell_28/MatMul_1/ReadVariableOp^gru_cell_28/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2F
!gru_cell_28/MatMul/ReadVariableOp!gru_cell_28/MatMul/ReadVariableOp2J
#gru_cell_28/MatMul_1/ReadVariableOp#gru_cell_28/MatMul_1/ReadVariableOp28
gru_cell_28/ReadVariableOpgru_cell_28/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
)__inference_gru_28_layer_call_fn_32656401
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_gru_28_layer_call_and_return_conditional_losses_326550272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�z
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655862

inputs.
*gru_28_gru_cell_28_readvariableop_resource5
1gru_28_gru_cell_28_matmul_readvariableop_resource7
3gru_28_gru_cell_28_matmul_1_readvariableop_resource,
(dense_156_matmul_readvariableop_resource-
)dense_156_biasadd_readvariableop_resource,
(dense_157_matmul_readvariableop_resource-
)dense_157_biasadd_readvariableop_resource
identity�� dense_156/BiasAdd/ReadVariableOp�dense_156/MatMul/ReadVariableOp� dense_157/BiasAdd/ReadVariableOp�dense_157/MatMul/ReadVariableOp�(gru_28/gru_cell_28/MatMul/ReadVariableOp�*gru_28/gru_cell_28/MatMul_1/ReadVariableOp�!gru_28/gru_cell_28/ReadVariableOp�gru_28/whileR
gru_28/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_28/Shape�
gru_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice/stack�
gru_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_1�
gru_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_2�
gru_28/strided_sliceStridedSlicegru_28/Shape:output:0#gru_28/strided_slice/stack:output:0%gru_28/strided_slice/stack_1:output:0%gru_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slicej
gru_28/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
gru_28/zeros/mul/y�
gru_28/zeros/mulMulgru_28/strided_slice:output:0gru_28/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/mulm
gru_28/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
gru_28/zeros/Less/y�
gru_28/zeros/LessLessgru_28/zeros/mul:z:0gru_28/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/Lessp
gru_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
gru_28/zeros/packed/1�
gru_28/zeros/packedPackgru_28/strided_slice:output:0gru_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_28/zeros/packedm
gru_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_28/zeros/Const�
gru_28/zerosFillgru_28/zeros/packed:output:0gru_28/zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
gru_28/zeros�
gru_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose/perm�
gru_28/transpose	Transposeinputsgru_28/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
gru_28/transposed
gru_28/Shape_1Shapegru_28/transpose:y:0*
T0*
_output_shapes
:2
gru_28/Shape_1�
gru_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_1/stack�
gru_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_1�
gru_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_2�
gru_28/strided_slice_1StridedSlicegru_28/Shape_1:output:0%gru_28/strided_slice_1/stack:output:0'gru_28/strided_slice_1/stack_1:output:0'gru_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slice_1�
"gru_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_28/TensorArrayV2/element_shape�
gru_28/TensorArrayV2TensorListReserve+gru_28/TensorArrayV2/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2�
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape�
.gru_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_28/transpose:y:0Egru_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_28/TensorArrayUnstack/TensorListFromTensor�
gru_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_2/stack�
gru_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_1�
gru_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_2�
gru_28/strided_slice_2StridedSlicegru_28/transpose:y:0%gru_28/strided_slice_2/stack:output:0'gru_28/strided_slice_2/stack_1:output:0'gru_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
gru_28/strided_slice_2�
!gru_28/gru_cell_28/ReadVariableOpReadVariableOp*gru_28_gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_28/gru_cell_28/ReadVariableOp�
gru_28/gru_cell_28/unstackUnpack)gru_28/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_28/gru_cell_28/unstack�
(gru_28/gru_cell_28/MatMul/ReadVariableOpReadVariableOp1gru_28_gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02*
(gru_28/gru_cell_28/MatMul/ReadVariableOp�
gru_28/gru_cell_28/MatMulMatMulgru_28/strided_slice_2:output:00gru_28/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/MatMul�
gru_28/gru_cell_28/BiasAddBiasAdd#gru_28/gru_cell_28/MatMul:product:0#gru_28/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/BiasAddv
gru_28/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/gru_cell_28/Const�
"gru_28/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_28/gru_cell_28/split/split_dim�
gru_28/gru_cell_28/splitSplit+gru_28/gru_cell_28/split/split_dim:output:0#gru_28/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_28/gru_cell_28/split�
*gru_28/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp3gru_28_gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02,
*gru_28/gru_cell_28/MatMul_1/ReadVariableOp�
gru_28/gru_cell_28/MatMul_1MatMulgru_28/zeros:output:02gru_28/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/MatMul_1�
gru_28/gru_cell_28/BiasAdd_1BiasAdd%gru_28/gru_cell_28/MatMul_1:product:0#gru_28/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/BiasAdd_1�
gru_28/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_28/gru_cell_28/Const_1�
$gru_28/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$gru_28/gru_cell_28/split_1/split_dim�
gru_28/gru_cell_28/split_1SplitV%gru_28/gru_cell_28/BiasAdd_1:output:0#gru_28/gru_cell_28/Const_1:output:0-gru_28/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_28/gru_cell_28/split_1�
gru_28/gru_cell_28/addAddV2!gru_28/gru_cell_28/split:output:0#gru_28/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add�
gru_28/gru_cell_28/SigmoidSigmoidgru_28/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/Sigmoid�
gru_28/gru_cell_28/add_1AddV2!gru_28/gru_cell_28/split:output:1#gru_28/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add_1�
gru_28/gru_cell_28/Sigmoid_1Sigmoidgru_28/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/Sigmoid_1�
gru_28/gru_cell_28/mulMul gru_28/gru_cell_28/Sigmoid_1:y:0#gru_28/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/mul�
gru_28/gru_cell_28/add_2AddV2!gru_28/gru_cell_28/split:output:2gru_28/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add_2�
gru_28/gru_cell_28/ReluRelugru_28/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/Relu�
gru_28/gru_cell_28/mul_1Mulgru_28/gru_cell_28/Sigmoid:y:0gru_28/zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/mul_1y
gru_28/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_28/gru_cell_28/sub/x�
gru_28/gru_cell_28/subSub!gru_28/gru_cell_28/sub/x:output:0gru_28/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/sub�
gru_28/gru_cell_28/mul_2Mulgru_28/gru_cell_28/sub:z:0%gru_28/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/mul_2�
gru_28/gru_cell_28/add_3AddV2gru_28/gru_cell_28/mul_1:z:0gru_28/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add_3�
$gru_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2&
$gru_28/TensorArrayV2_1/element_shape�
gru_28/TensorArrayV2_1TensorListReserve-gru_28/TensorArrayV2_1/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2_1\
gru_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/time�
gru_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
gru_28/while/maximum_iterationsx
gru_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/while/loop_counter�
gru_28/whileWhile"gru_28/while/loop_counter:output:0(gru_28/while/maximum_iterations:output:0gru_28/time:output:0gru_28/TensorArrayV2_1:handle:0gru_28/zeros:output:0gru_28/strided_slice_1:output:0>gru_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_28_gru_cell_28_readvariableop_resource1gru_28_gru_cell_28_matmul_readvariableop_resource3gru_28_gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*&
bodyR
gru_28_while_body_32655759*&
condR
gru_28_while_cond_32655758*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
gru_28/while�
7gru_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   29
7gru_28/TensorArrayV2Stack/TensorListStack/element_shape�
)gru_28/TensorArrayV2Stack/TensorListStackTensorListStackgru_28/while:output:3@gru_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02+
)gru_28/TensorArrayV2Stack/TensorListStack�
gru_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
gru_28/strided_slice_3/stack�
gru_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_28/strided_slice_3/stack_1�
gru_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_3/stack_2�
gru_28/strided_slice_3StridedSlice2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0%gru_28/strided_slice_3/stack:output:0'gru_28/strided_slice_3/stack_1:output:0'gru_28/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
gru_28/strided_slice_3�
gru_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose_1/perm�
gru_28/transpose_1	Transpose2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0 gru_28/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
gru_28/transpose_1t
gru_28/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_28/runtime�
dense_156/MatMul/ReadVariableOpReadVariableOp(dense_156_matmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02!
dense_156/MatMul/ReadVariableOp�
dense_156/MatMulMatMulgru_28/strided_slice_3:output:0'dense_156/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_156/MatMul�
 dense_156/BiasAdd/ReadVariableOpReadVariableOp)dense_156_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_156/BiasAdd/ReadVariableOp�
dense_156/BiasAddBiasAdddense_156/MatMul:product:0(dense_156/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_156/BiasAddv
dense_156/ReluReludense_156/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_156/Relu�
dense_157/MatMul/ReadVariableOpReadVariableOp(dense_157_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_157/MatMul/ReadVariableOp�
dense_157/MatMulMatMuldense_156/Relu:activations:0'dense_157/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_157/MatMul�
 dense_157/BiasAdd/ReadVariableOpReadVariableOp)dense_157_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_157/BiasAdd/ReadVariableOp�
dense_157/BiasAddBiasAdddense_157/MatMul:product:0(dense_157/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_157/BiasAdd�
IdentityIdentitydense_157/BiasAdd:output:0!^dense_156/BiasAdd/ReadVariableOp ^dense_156/MatMul/ReadVariableOp!^dense_157/BiasAdd/ReadVariableOp ^dense_157/MatMul/ReadVariableOp)^gru_28/gru_cell_28/MatMul/ReadVariableOp+^gru_28/gru_cell_28/MatMul_1/ReadVariableOp"^gru_28/gru_cell_28/ReadVariableOp^gru_28/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2D
 dense_156/BiasAdd/ReadVariableOp dense_156/BiasAdd/ReadVariableOp2B
dense_156/MatMul/ReadVariableOpdense_156/MatMul/ReadVariableOp2D
 dense_157/BiasAdd/ReadVariableOp dense_157/BiasAdd/ReadVariableOp2B
dense_157/MatMul/ReadVariableOpdense_157/MatMul/ReadVariableOp2T
(gru_28/gru_cell_28/MatMul/ReadVariableOp(gru_28/gru_cell_28/MatMul/ReadVariableOp2X
*gru_28/gru_cell_28/MatMul_1/ReadVariableOp*gru_28/gru_cell_28/MatMul_1/ReadVariableOp2F
!gru_28/gru_cell_28/ReadVariableOp!gru_28/gru_cell_28/ReadVariableOp2
gru_28/whilegru_28/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_32655225
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32655225___redundant_placeholder06
2while_while_cond_32655225___redundant_placeholder16
2while_while_cond_32655225___redundant_placeholder26
2while_while_cond_32655225___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�	
�
G__inference_dense_157_layer_call_and_return_conditional_losses_32656782

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�=
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32655145

inputs
gru_cell_28_32655069
gru_cell_28_32655071
gru_cell_28_32655073
identity��#gru_cell_28/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
#gru_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_28_32655069gru_cell_28_32655071gru_cell_28_32655073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_326547042%
#gru_cell_28/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_28_32655069gru_cell_28_32655071gru_cell_28_32655073*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32655081*
condR
while_cond_32655080*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0$^gru_cell_28/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2J
#gru_cell_28/StatefulPartitionedCall#gru_cell_28/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
.__inference_gru_cell_28_layer_call_fn_32656899

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_326547042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�=
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32655027

inputs
gru_cell_28_32654951
gru_cell_28_32654953
gru_cell_28_32654955
identity��#gru_cell_28/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
#gru_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_28_32654951gru_cell_28_32654953gru_cell_28_32654955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_326546642%
#gru_cell_28/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_28_32654951gru_cell_28_32654953gru_cell_28_32654955*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32654963*
condR
while_cond_32654962*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0$^gru_cell_28/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2J
#gru_cell_28/StatefulPartitionedCall#gru_cell_28/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
gru_28_while_cond_32655930*
&gru_28_while_gru_28_while_loop_counter0
,gru_28_while_gru_28_while_maximum_iterations
gru_28_while_placeholder
gru_28_while_placeholder_1
gru_28_while_placeholder_2,
(gru_28_while_less_gru_28_strided_slice_1D
@gru_28_while_gru_28_while_cond_32655930___redundant_placeholder0D
@gru_28_while_gru_28_while_cond_32655930___redundant_placeholder1D
@gru_28_while_gru_28_while_cond_32655930___redundant_placeholder2D
@gru_28_while_gru_28_while_cond_32655930___redundant_placeholder3
gru_28_while_identity
�
gru_28/while/LessLessgru_28_while_placeholder(gru_28_while_less_gru_28_strided_slice_1*
T0*
_output_shapes
: 2
gru_28/while/Lessr
gru_28/while/IdentityIdentitygru_28/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_28/while/Identity"7
gru_28_while_identitygru_28/while/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�[
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32655475

inputs'
#gru_cell_28_readvariableop_resource.
*gru_cell_28_matmul_readvariableop_resource0
,gru_cell_28_matmul_1_readvariableop_resource
identity��!gru_cell_28/MatMul/ReadVariableOp�#gru_cell_28/MatMul_1/ReadVariableOp�gru_cell_28/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_28/ReadVariableOpReadVariableOp#gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_28/ReadVariableOp�
gru_cell_28/unstackUnpack"gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_28/unstack�
!gru_cell_28/MatMul/ReadVariableOpReadVariableOp*gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_cell_28/MatMul/ReadVariableOp�
gru_cell_28/MatMulMatMulstrided_slice_2:output:0)gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul�
gru_cell_28/BiasAddBiasAddgru_cell_28/MatMul:product:0gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAddh
gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_28/Const�
gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split/split_dim�
gru_cell_28/splitSplit$gru_cell_28/split/split_dim:output:0gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split�
#gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02%
#gru_cell_28/MatMul_1/ReadVariableOp�
gru_cell_28/MatMul_1MatMulzeros:output:0+gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul_1�
gru_cell_28/BiasAdd_1BiasAddgru_cell_28/MatMul_1:product:0gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAdd_1
gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_28/Const_1�
gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split_1/split_dim�
gru_cell_28/split_1SplitVgru_cell_28/BiasAdd_1:output:0gru_cell_28/Const_1:output:0&gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split_1�
gru_cell_28/addAddV2gru_cell_28/split:output:0gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add|
gru_cell_28/SigmoidSigmoidgru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid�
gru_cell_28/add_1AddV2gru_cell_28/split:output:1gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_1�
gru_cell_28/Sigmoid_1Sigmoidgru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid_1�
gru_cell_28/mulMulgru_cell_28/Sigmoid_1:y:0gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul�
gru_cell_28/add_2AddV2gru_cell_28/split:output:2gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_2u
gru_cell_28/ReluRelugru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Relu�
gru_cell_28/mul_1Mulgru_cell_28/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_1k
gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_28/sub/x�
gru_cell_28/subSubgru_cell_28/sub/x:output:0gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/sub�
gru_cell_28/mul_2Mulgru_cell_28/sub:z:0gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_2�
gru_cell_28/add_3AddV2gru_cell_28/mul_1:z:0gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_28_readvariableop_resource*gru_cell_28_matmul_readvariableop_resource,gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32655385*
condR
while_cond_32655384*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^gru_cell_28/MatMul/ReadVariableOp$^gru_cell_28/MatMul_1/ReadVariableOp^gru_cell_28/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2F
!gru_cell_28/MatMul/ReadVariableOp!gru_cell_28/MatMul/ReadVariableOp2J
#gru_cell_28/MatMul_1/ReadVariableOp#gru_cell_28/MatMul_1/ReadVariableOp28
gru_cell_28/ReadVariableOpgru_cell_28/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�
while_body_32656640
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_28_readvariableop_resource_06
2while_gru_cell_28_matmul_readvariableop_resource_08
4while_gru_cell_28_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_28_readvariableop_resource4
0while_gru_cell_28_matmul_readvariableop_resource6
2while_gru_cell_28_matmul_1_readvariableop_resource��'while/gru_cell_28/MatMul/ReadVariableOp�)while/gru_cell_28/MatMul_1/ReadVariableOp� while/gru_cell_28/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
 while/gru_cell_28/ReadVariableOpReadVariableOp+while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02"
 while/gru_cell_28/ReadVariableOp�
while/gru_cell_28/unstackUnpack(while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_28/unstack�
'while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'while/gru_cell_28/MatMul/ReadVariableOp�
while/gru_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul�
while/gru_cell_28/BiasAddBiasAdd"while/gru_cell_28/MatMul:product:0"while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAddt
while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_28/Const�
!while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!while/gru_cell_28/split/split_dim�
while/gru_cell_28/splitSplit*while/gru_cell_28/split/split_dim:output:0"while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split�
)while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02+
)while/gru_cell_28/MatMul_1/ReadVariableOp�
while/gru_cell_28/MatMul_1MatMulwhile_placeholder_21while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul_1�
while/gru_cell_28/BiasAdd_1BiasAdd$while/gru_cell_28/MatMul_1:product:0"while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAdd_1�
while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_28/Const_1�
#while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#while/gru_cell_28/split_1/split_dim�
while/gru_cell_28/split_1SplitV$while/gru_cell_28/BiasAdd_1:output:0"while/gru_cell_28/Const_1:output:0,while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split_1�
while/gru_cell_28/addAddV2 while/gru_cell_28/split:output:0"while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add�
while/gru_cell_28/SigmoidSigmoidwhile/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid�
while/gru_cell_28/add_1AddV2 while/gru_cell_28/split:output:1"while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_1�
while/gru_cell_28/Sigmoid_1Sigmoidwhile/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid_1�
while/gru_cell_28/mulMulwhile/gru_cell_28/Sigmoid_1:y:0"while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul�
while/gru_cell_28/add_2AddV2 while/gru_cell_28/split:output:2while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_2�
while/gru_cell_28/ReluReluwhile/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Relu�
while/gru_cell_28/mul_1Mulwhile/gru_cell_28/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_1w
while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_28/sub/x�
while/gru_cell_28/subSub while/gru_cell_28/sub/x:output:0while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/sub�
while/gru_cell_28/mul_2Mulwhile/gru_cell_28/sub:z:0$while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_2�
while/gru_cell_28/add_3AddV2while/gru_cell_28/mul_1:z:0while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_3�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_28/add_3:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"j
2while_gru_cell_28_matmul_1_readvariableop_resource4while_gru_cell_28_matmul_1_readvariableop_resource_0"f
0while_gru_cell_28_matmul_readvariableop_resource2while_gru_cell_28_matmul_readvariableop_resource_0"X
)while_gru_cell_28_readvariableop_resource+while_gru_cell_28_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2R
'while/gru_cell_28/MatMul/ReadVariableOp'while/gru_cell_28/MatMul/ReadVariableOp2V
)while/gru_cell_28/MatMul_1/ReadVariableOp)while/gru_cell_28/MatMul_1/ReadVariableOp2D
 while/gru_cell_28/ReadVariableOp while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_32655081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0 
while_gru_cell_28_32655103_0 
while_gru_cell_28_32655105_0 
while_gru_cell_28_32655107_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_28_32655103
while_gru_cell_28_32655105
while_gru_cell_28_32655107��)while/gru_cell_28/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
)while/gru_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_28_32655103_0while_gru_cell_28_32655105_0while_gru_cell_28_32655107_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_326547042+
)while/gru_cell_28/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_28/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity2while/gru_cell_28/StatefulPartitionedCall:output:1*^while/gru_cell_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2
while/Identity_4":
while_gru_cell_28_32655103while_gru_cell_28_32655103_0":
while_gru_cell_28_32655105while_gru_cell_28_32655105_0":
while_gru_cell_28_32655107while_gru_cell_28_32655107_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2V
)while/gru_cell_28/StatefulPartitionedCall)while/gru_cell_28/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
��
�
#__inference__wrapped_model_32654592
gru_28_input<
8sequential_78_gru_28_gru_cell_28_readvariableop_resourceC
?sequential_78_gru_28_gru_cell_28_matmul_readvariableop_resourceE
Asequential_78_gru_28_gru_cell_28_matmul_1_readvariableop_resource:
6sequential_78_dense_156_matmul_readvariableop_resource;
7sequential_78_dense_156_biasadd_readvariableop_resource:
6sequential_78_dense_157_matmul_readvariableop_resource;
7sequential_78_dense_157_biasadd_readvariableop_resource
identity��.sequential_78/dense_156/BiasAdd/ReadVariableOp�-sequential_78/dense_156/MatMul/ReadVariableOp�.sequential_78/dense_157/BiasAdd/ReadVariableOp�-sequential_78/dense_157/MatMul/ReadVariableOp�6sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOp�8sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOp�/sequential_78/gru_28/gru_cell_28/ReadVariableOp�sequential_78/gru_28/whilet
sequential_78/gru_28/ShapeShapegru_28_input*
T0*
_output_shapes
:2
sequential_78/gru_28/Shape�
(sequential_78/gru_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_78/gru_28/strided_slice/stack�
*sequential_78/gru_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_78/gru_28/strided_slice/stack_1�
*sequential_78/gru_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_78/gru_28/strided_slice/stack_2�
"sequential_78/gru_28/strided_sliceStridedSlice#sequential_78/gru_28/Shape:output:01sequential_78/gru_28/strided_slice/stack:output:03sequential_78/gru_28/strided_slice/stack_1:output:03sequential_78/gru_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_78/gru_28/strided_slice�
 sequential_78/gru_28/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2"
 sequential_78/gru_28/zeros/mul/y�
sequential_78/gru_28/zeros/mulMul+sequential_78/gru_28/strided_slice:output:0)sequential_78/gru_28/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_78/gru_28/zeros/mul�
!sequential_78/gru_28/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!sequential_78/gru_28/zeros/Less/y�
sequential_78/gru_28/zeros/LessLess"sequential_78/gru_28/zeros/mul:z:0*sequential_78/gru_28/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_78/gru_28/zeros/Less�
#sequential_78/gru_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2%
#sequential_78/gru_28/zeros/packed/1�
!sequential_78/gru_28/zeros/packedPack+sequential_78/gru_28/strided_slice:output:0,sequential_78/gru_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_78/gru_28/zeros/packed�
 sequential_78/gru_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_78/gru_28/zeros/Const�
sequential_78/gru_28/zerosFill*sequential_78/gru_28/zeros/packed:output:0)sequential_78/gru_28/zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
sequential_78/gru_28/zeros�
#sequential_78/gru_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_78/gru_28/transpose/perm�
sequential_78/gru_28/transpose	Transposegru_28_input,sequential_78/gru_28/transpose/perm:output:0*
T0*+
_output_shapes
:���������2 
sequential_78/gru_28/transpose�
sequential_78/gru_28/Shape_1Shape"sequential_78/gru_28/transpose:y:0*
T0*
_output_shapes
:2
sequential_78/gru_28/Shape_1�
*sequential_78/gru_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_78/gru_28/strided_slice_1/stack�
,sequential_78/gru_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_78/gru_28/strided_slice_1/stack_1�
,sequential_78/gru_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_78/gru_28/strided_slice_1/stack_2�
$sequential_78/gru_28/strided_slice_1StridedSlice%sequential_78/gru_28/Shape_1:output:03sequential_78/gru_28/strided_slice_1/stack:output:05sequential_78/gru_28/strided_slice_1/stack_1:output:05sequential_78/gru_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_78/gru_28/strided_slice_1�
0sequential_78/gru_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_78/gru_28/TensorArrayV2/element_shape�
"sequential_78/gru_28/TensorArrayV2TensorListReserve9sequential_78/gru_28/TensorArrayV2/element_shape:output:0-sequential_78/gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_78/gru_28/TensorArrayV2�
Jsequential_78/gru_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jsequential_78/gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape�
<sequential_78/gru_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_78/gru_28/transpose:y:0Ssequential_78/gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_78/gru_28/TensorArrayUnstack/TensorListFromTensor�
*sequential_78/gru_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_78/gru_28/strided_slice_2/stack�
,sequential_78/gru_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_78/gru_28/strided_slice_2/stack_1�
,sequential_78/gru_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_78/gru_28/strided_slice_2/stack_2�
$sequential_78/gru_28/strided_slice_2StridedSlice"sequential_78/gru_28/transpose:y:03sequential_78/gru_28/strided_slice_2/stack:output:05sequential_78/gru_28/strided_slice_2/stack_1:output:05sequential_78/gru_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2&
$sequential_78/gru_28/strided_slice_2�
/sequential_78/gru_28/gru_cell_28/ReadVariableOpReadVariableOp8sequential_78_gru_28_gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype021
/sequential_78/gru_28/gru_cell_28/ReadVariableOp�
(sequential_78/gru_28/gru_cell_28/unstackUnpack7sequential_78/gru_28/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2*
(sequential_78/gru_28/gru_cell_28/unstack�
6sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOpReadVariableOp?sequential_78_gru_28_gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype028
6sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOp�
'sequential_78/gru_28/gru_cell_28/MatMulMatMul-sequential_78/gru_28/strided_slice_2:output:0>sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'sequential_78/gru_28/gru_cell_28/MatMul�
(sequential_78/gru_28/gru_cell_28/BiasAddBiasAdd1sequential_78/gru_28/gru_cell_28/MatMul:product:01sequential_78/gru_28/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2*
(sequential_78/gru_28/gru_cell_28/BiasAdd�
&sequential_78/gru_28/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_78/gru_28/gru_cell_28/Const�
0sequential_78/gru_28/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_78/gru_28/gru_cell_28/split/split_dim�
&sequential_78/gru_28/gru_cell_28/splitSplit9sequential_78/gru_28/gru_cell_28/split/split_dim:output:01sequential_78/gru_28/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2(
&sequential_78/gru_28/gru_cell_28/split�
8sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOpAsequential_78_gru_28_gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02:
8sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOp�
)sequential_78/gru_28/gru_cell_28/MatMul_1MatMul#sequential_78/gru_28/zeros:output:0@sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_78/gru_28/gru_cell_28/MatMul_1�
*sequential_78/gru_28/gru_cell_28/BiasAdd_1BiasAdd3sequential_78/gru_28/gru_cell_28/MatMul_1:product:01sequential_78/gru_28/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2,
*sequential_78/gru_28/gru_cell_28/BiasAdd_1�
(sequential_78/gru_28/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2*
(sequential_78/gru_28/gru_cell_28/Const_1�
2sequential_78/gru_28/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2sequential_78/gru_28/gru_cell_28/split_1/split_dim�
(sequential_78/gru_28/gru_cell_28/split_1SplitV3sequential_78/gru_28/gru_cell_28/BiasAdd_1:output:01sequential_78/gru_28/gru_cell_28/Const_1:output:0;sequential_78/gru_28/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2*
(sequential_78/gru_28/gru_cell_28/split_1�
$sequential_78/gru_28/gru_cell_28/addAddV2/sequential_78/gru_28/gru_cell_28/split:output:01sequential_78/gru_28/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2&
$sequential_78/gru_28/gru_cell_28/add�
(sequential_78/gru_28/gru_cell_28/SigmoidSigmoid(sequential_78/gru_28/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2*
(sequential_78/gru_28/gru_cell_28/Sigmoid�
&sequential_78/gru_28/gru_cell_28/add_1AddV2/sequential_78/gru_28/gru_cell_28/split:output:11sequential_78/gru_28/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2(
&sequential_78/gru_28/gru_cell_28/add_1�
*sequential_78/gru_28/gru_cell_28/Sigmoid_1Sigmoid*sequential_78/gru_28/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2,
*sequential_78/gru_28/gru_cell_28/Sigmoid_1�
$sequential_78/gru_28/gru_cell_28/mulMul.sequential_78/gru_28/gru_cell_28/Sigmoid_1:y:01sequential_78/gru_28/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2&
$sequential_78/gru_28/gru_cell_28/mul�
&sequential_78/gru_28/gru_cell_28/add_2AddV2/sequential_78/gru_28/gru_cell_28/split:output:2(sequential_78/gru_28/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2(
&sequential_78/gru_28/gru_cell_28/add_2�
%sequential_78/gru_28/gru_cell_28/ReluRelu*sequential_78/gru_28/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2'
%sequential_78/gru_28/gru_cell_28/Relu�
&sequential_78/gru_28/gru_cell_28/mul_1Mul,sequential_78/gru_28/gru_cell_28/Sigmoid:y:0#sequential_78/gru_28/zeros:output:0*
T0*'
_output_shapes
:���������K2(
&sequential_78/gru_28/gru_cell_28/mul_1�
&sequential_78/gru_28/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2(
&sequential_78/gru_28/gru_cell_28/sub/x�
$sequential_78/gru_28/gru_cell_28/subSub/sequential_78/gru_28/gru_cell_28/sub/x:output:0,sequential_78/gru_28/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2&
$sequential_78/gru_28/gru_cell_28/sub�
&sequential_78/gru_28/gru_cell_28/mul_2Mul(sequential_78/gru_28/gru_cell_28/sub:z:03sequential_78/gru_28/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2(
&sequential_78/gru_28/gru_cell_28/mul_2�
&sequential_78/gru_28/gru_cell_28/add_3AddV2*sequential_78/gru_28/gru_cell_28/mul_1:z:0*sequential_78/gru_28/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2(
&sequential_78/gru_28/gru_cell_28/add_3�
2sequential_78/gru_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   24
2sequential_78/gru_28/TensorArrayV2_1/element_shape�
$sequential_78/gru_28/TensorArrayV2_1TensorListReserve;sequential_78/gru_28/TensorArrayV2_1/element_shape:output:0-sequential_78/gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_78/gru_28/TensorArrayV2_1x
sequential_78/gru_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_78/gru_28/time�
-sequential_78/gru_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_78/gru_28/while/maximum_iterations�
'sequential_78/gru_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_78/gru_28/while/loop_counter�
sequential_78/gru_28/whileWhile0sequential_78/gru_28/while/loop_counter:output:06sequential_78/gru_28/while/maximum_iterations:output:0"sequential_78/gru_28/time:output:0-sequential_78/gru_28/TensorArrayV2_1:handle:0#sequential_78/gru_28/zeros:output:0-sequential_78/gru_28/strided_slice_1:output:0Lsequential_78/gru_28/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_78_gru_28_gru_cell_28_readvariableop_resource?sequential_78_gru_28_gru_cell_28_matmul_readvariableop_resourceAsequential_78_gru_28_gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*4
body,R*
(sequential_78_gru_28_while_body_32654489*4
cond,R*
(sequential_78_gru_28_while_cond_32654488*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
sequential_78/gru_28/while�
Esequential_78/gru_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2G
Esequential_78/gru_28/TensorArrayV2Stack/TensorListStack/element_shape�
7sequential_78/gru_28/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_78/gru_28/while:output:3Nsequential_78/gru_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype029
7sequential_78/gru_28/TensorArrayV2Stack/TensorListStack�
*sequential_78/gru_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*sequential_78/gru_28/strided_slice_3/stack�
,sequential_78/gru_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_78/gru_28/strided_slice_3/stack_1�
,sequential_78/gru_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_78/gru_28/strided_slice_3/stack_2�
$sequential_78/gru_28/strided_slice_3StridedSlice@sequential_78/gru_28/TensorArrayV2Stack/TensorListStack:tensor:03sequential_78/gru_28/strided_slice_3/stack:output:05sequential_78/gru_28/strided_slice_3/stack_1:output:05sequential_78/gru_28/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2&
$sequential_78/gru_28/strided_slice_3�
%sequential_78/gru_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_78/gru_28/transpose_1/perm�
 sequential_78/gru_28/transpose_1	Transpose@sequential_78/gru_28/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_78/gru_28/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2"
 sequential_78/gru_28/transpose_1�
sequential_78/gru_28/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_78/gru_28/runtime�
-sequential_78/dense_156/MatMul/ReadVariableOpReadVariableOp6sequential_78_dense_156_matmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02/
-sequential_78/dense_156/MatMul/ReadVariableOp�
sequential_78/dense_156/MatMulMatMul-sequential_78/gru_28/strided_slice_3:output:05sequential_78/dense_156/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2 
sequential_78/dense_156/MatMul�
.sequential_78/dense_156/BiasAdd/ReadVariableOpReadVariableOp7sequential_78_dense_156_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_78/dense_156/BiasAdd/ReadVariableOp�
sequential_78/dense_156/BiasAddBiasAdd(sequential_78/dense_156/MatMul:product:06sequential_78/dense_156/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2!
sequential_78/dense_156/BiasAdd�
sequential_78/dense_156/ReluRelu(sequential_78/dense_156/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential_78/dense_156/Relu�
-sequential_78/dense_157/MatMul/ReadVariableOpReadVariableOp6sequential_78_dense_157_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_78/dense_157/MatMul/ReadVariableOp�
sequential_78/dense_157/MatMulMatMul*sequential_78/dense_156/Relu:activations:05sequential_78/dense_157/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_78/dense_157/MatMul�
.sequential_78/dense_157/BiasAdd/ReadVariableOpReadVariableOp7sequential_78_dense_157_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_78/dense_157/BiasAdd/ReadVariableOp�
sequential_78/dense_157/BiasAddBiasAdd(sequential_78/dense_157/MatMul:product:06sequential_78/dense_157/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
sequential_78/dense_157/BiasAdd�
IdentityIdentity(sequential_78/dense_157/BiasAdd:output:0/^sequential_78/dense_156/BiasAdd/ReadVariableOp.^sequential_78/dense_156/MatMul/ReadVariableOp/^sequential_78/dense_157/BiasAdd/ReadVariableOp.^sequential_78/dense_157/MatMul/ReadVariableOp7^sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOp9^sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOp0^sequential_78/gru_28/gru_cell_28/ReadVariableOp^sequential_78/gru_28/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2`
.sequential_78/dense_156/BiasAdd/ReadVariableOp.sequential_78/dense_156/BiasAdd/ReadVariableOp2^
-sequential_78/dense_156/MatMul/ReadVariableOp-sequential_78/dense_156/MatMul/ReadVariableOp2`
.sequential_78/dense_157/BiasAdd/ReadVariableOp.sequential_78/dense_157/BiasAdd/ReadVariableOp2^
-sequential_78/dense_157/MatMul/ReadVariableOp-sequential_78/dense_157/MatMul/ReadVariableOp2p
6sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOp6sequential_78/gru_28/gru_cell_28/MatMul/ReadVariableOp2t
8sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOp8sequential_78/gru_28/gru_cell_28/MatMul_1/ReadVariableOp2b
/sequential_78/gru_28/gru_cell_28/ReadVariableOp/sequential_78/gru_28/gru_cell_28/ReadVariableOp28
sequential_78/gru_28/whilesequential_78/gru_28/while:Y U
+
_output_shapes
:���������
&
_user_specified_namegru_28_input
�z
�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32656034

inputs.
*gru_28_gru_cell_28_readvariableop_resource5
1gru_28_gru_cell_28_matmul_readvariableop_resource7
3gru_28_gru_cell_28_matmul_1_readvariableop_resource,
(dense_156_matmul_readvariableop_resource-
)dense_156_biasadd_readvariableop_resource,
(dense_157_matmul_readvariableop_resource-
)dense_157_biasadd_readvariableop_resource
identity�� dense_156/BiasAdd/ReadVariableOp�dense_156/MatMul/ReadVariableOp� dense_157/BiasAdd/ReadVariableOp�dense_157/MatMul/ReadVariableOp�(gru_28/gru_cell_28/MatMul/ReadVariableOp�*gru_28/gru_cell_28/MatMul_1/ReadVariableOp�!gru_28/gru_cell_28/ReadVariableOp�gru_28/whileR
gru_28/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_28/Shape�
gru_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice/stack�
gru_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_1�
gru_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_2�
gru_28/strided_sliceStridedSlicegru_28/Shape:output:0#gru_28/strided_slice/stack:output:0%gru_28/strided_slice/stack_1:output:0%gru_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slicej
gru_28/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
gru_28/zeros/mul/y�
gru_28/zeros/mulMulgru_28/strided_slice:output:0gru_28/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/mulm
gru_28/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
gru_28/zeros/Less/y�
gru_28/zeros/LessLessgru_28/zeros/mul:z:0gru_28/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/Lessp
gru_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
gru_28/zeros/packed/1�
gru_28/zeros/packedPackgru_28/strided_slice:output:0gru_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_28/zeros/packedm
gru_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_28/zeros/Const�
gru_28/zerosFillgru_28/zeros/packed:output:0gru_28/zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
gru_28/zeros�
gru_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose/perm�
gru_28/transpose	Transposeinputsgru_28/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
gru_28/transposed
gru_28/Shape_1Shapegru_28/transpose:y:0*
T0*
_output_shapes
:2
gru_28/Shape_1�
gru_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_1/stack�
gru_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_1�
gru_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_2�
gru_28/strided_slice_1StridedSlicegru_28/Shape_1:output:0%gru_28/strided_slice_1/stack:output:0'gru_28/strided_slice_1/stack_1:output:0'gru_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slice_1�
"gru_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_28/TensorArrayV2/element_shape�
gru_28/TensorArrayV2TensorListReserve+gru_28/TensorArrayV2/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2�
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape�
.gru_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_28/transpose:y:0Egru_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_28/TensorArrayUnstack/TensorListFromTensor�
gru_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_2/stack�
gru_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_1�
gru_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_2�
gru_28/strided_slice_2StridedSlicegru_28/transpose:y:0%gru_28/strided_slice_2/stack:output:0'gru_28/strided_slice_2/stack_1:output:0'gru_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
gru_28/strided_slice_2�
!gru_28/gru_cell_28/ReadVariableOpReadVariableOp*gru_28_gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_28/gru_cell_28/ReadVariableOp�
gru_28/gru_cell_28/unstackUnpack)gru_28/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_28/gru_cell_28/unstack�
(gru_28/gru_cell_28/MatMul/ReadVariableOpReadVariableOp1gru_28_gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02*
(gru_28/gru_cell_28/MatMul/ReadVariableOp�
gru_28/gru_cell_28/MatMulMatMulgru_28/strided_slice_2:output:00gru_28/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/MatMul�
gru_28/gru_cell_28/BiasAddBiasAdd#gru_28/gru_cell_28/MatMul:product:0#gru_28/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/BiasAddv
gru_28/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/gru_cell_28/Const�
"gru_28/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_28/gru_cell_28/split/split_dim�
gru_28/gru_cell_28/splitSplit+gru_28/gru_cell_28/split/split_dim:output:0#gru_28/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_28/gru_cell_28/split�
*gru_28/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp3gru_28_gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02,
*gru_28/gru_cell_28/MatMul_1/ReadVariableOp�
gru_28/gru_cell_28/MatMul_1MatMulgru_28/zeros:output:02gru_28/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/MatMul_1�
gru_28/gru_cell_28/BiasAdd_1BiasAdd%gru_28/gru_cell_28/MatMul_1:product:0#gru_28/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_28/gru_cell_28/BiasAdd_1�
gru_28/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_28/gru_cell_28/Const_1�
$gru_28/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$gru_28/gru_cell_28/split_1/split_dim�
gru_28/gru_cell_28/split_1SplitV%gru_28/gru_cell_28/BiasAdd_1:output:0#gru_28/gru_cell_28/Const_1:output:0-gru_28/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_28/gru_cell_28/split_1�
gru_28/gru_cell_28/addAddV2!gru_28/gru_cell_28/split:output:0#gru_28/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add�
gru_28/gru_cell_28/SigmoidSigmoidgru_28/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/Sigmoid�
gru_28/gru_cell_28/add_1AddV2!gru_28/gru_cell_28/split:output:1#gru_28/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add_1�
gru_28/gru_cell_28/Sigmoid_1Sigmoidgru_28/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/Sigmoid_1�
gru_28/gru_cell_28/mulMul gru_28/gru_cell_28/Sigmoid_1:y:0#gru_28/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/mul�
gru_28/gru_cell_28/add_2AddV2!gru_28/gru_cell_28/split:output:2gru_28/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add_2�
gru_28/gru_cell_28/ReluRelugru_28/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/Relu�
gru_28/gru_cell_28/mul_1Mulgru_28/gru_cell_28/Sigmoid:y:0gru_28/zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/mul_1y
gru_28/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_28/gru_cell_28/sub/x�
gru_28/gru_cell_28/subSub!gru_28/gru_cell_28/sub/x:output:0gru_28/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/sub�
gru_28/gru_cell_28/mul_2Mulgru_28/gru_cell_28/sub:z:0%gru_28/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/mul_2�
gru_28/gru_cell_28/add_3AddV2gru_28/gru_cell_28/mul_1:z:0gru_28/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_28/gru_cell_28/add_3�
$gru_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2&
$gru_28/TensorArrayV2_1/element_shape�
gru_28/TensorArrayV2_1TensorListReserve-gru_28/TensorArrayV2_1/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2_1\
gru_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/time�
gru_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
gru_28/while/maximum_iterationsx
gru_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/while/loop_counter�
gru_28/whileWhile"gru_28/while/loop_counter:output:0(gru_28/while/maximum_iterations:output:0gru_28/time:output:0gru_28/TensorArrayV2_1:handle:0gru_28/zeros:output:0gru_28/strided_slice_1:output:0>gru_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_28_gru_cell_28_readvariableop_resource1gru_28_gru_cell_28_matmul_readvariableop_resource3gru_28_gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*&
bodyR
gru_28_while_body_32655931*&
condR
gru_28_while_cond_32655930*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
gru_28/while�
7gru_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   29
7gru_28/TensorArrayV2Stack/TensorListStack/element_shape�
)gru_28/TensorArrayV2Stack/TensorListStackTensorListStackgru_28/while:output:3@gru_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02+
)gru_28/TensorArrayV2Stack/TensorListStack�
gru_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
gru_28/strided_slice_3/stack�
gru_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_28/strided_slice_3/stack_1�
gru_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_3/stack_2�
gru_28/strided_slice_3StridedSlice2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0%gru_28/strided_slice_3/stack:output:0'gru_28/strided_slice_3/stack_1:output:0'gru_28/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
gru_28/strided_slice_3�
gru_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose_1/perm�
gru_28/transpose_1	Transpose2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0 gru_28/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
gru_28/transpose_1t
gru_28/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_28/runtime�
dense_156/MatMul/ReadVariableOpReadVariableOp(dense_156_matmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02!
dense_156/MatMul/ReadVariableOp�
dense_156/MatMulMatMulgru_28/strided_slice_3:output:0'dense_156/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_156/MatMul�
 dense_156/BiasAdd/ReadVariableOpReadVariableOp)dense_156_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_156/BiasAdd/ReadVariableOp�
dense_156/BiasAddBiasAdddense_156/MatMul:product:0(dense_156/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_156/BiasAddv
dense_156/ReluReludense_156/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_156/Relu�
dense_157/MatMul/ReadVariableOpReadVariableOp(dense_157_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_157/MatMul/ReadVariableOp�
dense_157/MatMulMatMuldense_156/Relu:activations:0'dense_157/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_157/MatMul�
 dense_157/BiasAdd/ReadVariableOpReadVariableOp)dense_157_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_157/BiasAdd/ReadVariableOp�
dense_157/BiasAddBiasAdddense_157/MatMul:product:0(dense_157/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_157/BiasAdd�
IdentityIdentitydense_157/BiasAdd:output:0!^dense_156/BiasAdd/ReadVariableOp ^dense_156/MatMul/ReadVariableOp!^dense_157/BiasAdd/ReadVariableOp ^dense_157/MatMul/ReadVariableOp)^gru_28/gru_cell_28/MatMul/ReadVariableOp+^gru_28/gru_cell_28/MatMul_1/ReadVariableOp"^gru_28/gru_cell_28/ReadVariableOp^gru_28/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2D
 dense_156/BiasAdd/ReadVariableOp dense_156/BiasAdd/ReadVariableOp2B
dense_156/MatMul/ReadVariableOpdense_156/MatMul/ReadVariableOp2D
 dense_157/BiasAdd/ReadVariableOp dense_157/BiasAdd/ReadVariableOp2B
dense_157/MatMul/ReadVariableOpdense_157/MatMul/ReadVariableOp2T
(gru_28/gru_cell_28/MatMul/ReadVariableOp(gru_28/gru_cell_28/MatMul/ReadVariableOp2X
*gru_28/gru_cell_28/MatMul_1/ReadVariableOp*gru_28/gru_cell_28/MatMul_1/ReadVariableOp2F
!gru_28/gru_cell_28/ReadVariableOp!gru_28/gru_cell_28/ReadVariableOp2
gru_28/whilegru_28/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�
while_body_32656481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_28_readvariableop_resource_06
2while_gru_cell_28_matmul_readvariableop_resource_08
4while_gru_cell_28_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_28_readvariableop_resource4
0while_gru_cell_28_matmul_readvariableop_resource6
2while_gru_cell_28_matmul_1_readvariableop_resource��'while/gru_cell_28/MatMul/ReadVariableOp�)while/gru_cell_28/MatMul_1/ReadVariableOp� while/gru_cell_28/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
 while/gru_cell_28/ReadVariableOpReadVariableOp+while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02"
 while/gru_cell_28/ReadVariableOp�
while/gru_cell_28/unstackUnpack(while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_28/unstack�
'while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'while/gru_cell_28/MatMul/ReadVariableOp�
while/gru_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul�
while/gru_cell_28/BiasAddBiasAdd"while/gru_cell_28/MatMul:product:0"while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAddt
while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_28/Const�
!while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!while/gru_cell_28/split/split_dim�
while/gru_cell_28/splitSplit*while/gru_cell_28/split/split_dim:output:0"while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split�
)while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02+
)while/gru_cell_28/MatMul_1/ReadVariableOp�
while/gru_cell_28/MatMul_1MatMulwhile_placeholder_21while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul_1�
while/gru_cell_28/BiasAdd_1BiasAdd$while/gru_cell_28/MatMul_1:product:0"while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAdd_1�
while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_28/Const_1�
#while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#while/gru_cell_28/split_1/split_dim�
while/gru_cell_28/split_1SplitV$while/gru_cell_28/BiasAdd_1:output:0"while/gru_cell_28/Const_1:output:0,while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split_1�
while/gru_cell_28/addAddV2 while/gru_cell_28/split:output:0"while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add�
while/gru_cell_28/SigmoidSigmoidwhile/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid�
while/gru_cell_28/add_1AddV2 while/gru_cell_28/split:output:1"while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_1�
while/gru_cell_28/Sigmoid_1Sigmoidwhile/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid_1�
while/gru_cell_28/mulMulwhile/gru_cell_28/Sigmoid_1:y:0"while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul�
while/gru_cell_28/add_2AddV2 while/gru_cell_28/split:output:2while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_2�
while/gru_cell_28/ReluReluwhile/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Relu�
while/gru_cell_28/mul_1Mulwhile/gru_cell_28/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_1w
while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_28/sub/x�
while/gru_cell_28/subSub while/gru_cell_28/sub/x:output:0while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/sub�
while/gru_cell_28/mul_2Mulwhile/gru_cell_28/sub:z:0$while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_2�
while/gru_cell_28/add_3AddV2while/gru_cell_28/mul_1:z:0while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_3�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_28/add_3:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"j
2while_gru_cell_28_matmul_1_readvariableop_resource4while_gru_cell_28_matmul_1_readvariableop_resource_0"f
0while_gru_cell_28_matmul_readvariableop_resource2while_gru_cell_28_matmul_readvariableop_resource_0"X
)while_gru_cell_28_readvariableop_resource+while_gru_cell_28_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2R
'while/gru_cell_28/MatMul/ReadVariableOp'while/gru_cell_28/MatMul/ReadVariableOp2V
)while/gru_cell_28/MatMul_1/ReadVariableOp)while/gru_cell_28/MatMul_1/ReadVariableOp2D
 while/gru_cell_28/ReadVariableOp while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
(sequential_78_gru_28_while_cond_32654488F
Bsequential_78_gru_28_while_sequential_78_gru_28_while_loop_counterL
Hsequential_78_gru_28_while_sequential_78_gru_28_while_maximum_iterations*
&sequential_78_gru_28_while_placeholder,
(sequential_78_gru_28_while_placeholder_1,
(sequential_78_gru_28_while_placeholder_2H
Dsequential_78_gru_28_while_less_sequential_78_gru_28_strided_slice_1`
\sequential_78_gru_28_while_sequential_78_gru_28_while_cond_32654488___redundant_placeholder0`
\sequential_78_gru_28_while_sequential_78_gru_28_while_cond_32654488___redundant_placeholder1`
\sequential_78_gru_28_while_sequential_78_gru_28_while_cond_32654488___redundant_placeholder2`
\sequential_78_gru_28_while_sequential_78_gru_28_while_cond_32654488___redundant_placeholder3'
#sequential_78_gru_28_while_identity
�
sequential_78/gru_28/while/LessLess&sequential_78_gru_28_while_placeholderDsequential_78_gru_28_while_less_sequential_78_gru_28_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_78/gru_28/while/Less�
#sequential_78/gru_28/while/IdentityIdentity#sequential_78/gru_28/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_78/gru_28/while/Identity"S
#sequential_78_gru_28_while_identity,sequential_78/gru_28/while/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�G
�
while_body_32655385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_28_readvariableop_resource_06
2while_gru_cell_28_matmul_readvariableop_resource_08
4while_gru_cell_28_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_28_readvariableop_resource4
0while_gru_cell_28_matmul_readvariableop_resource6
2while_gru_cell_28_matmul_1_readvariableop_resource��'while/gru_cell_28/MatMul/ReadVariableOp�)while/gru_cell_28/MatMul_1/ReadVariableOp� while/gru_cell_28/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
 while/gru_cell_28/ReadVariableOpReadVariableOp+while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02"
 while/gru_cell_28/ReadVariableOp�
while/gru_cell_28/unstackUnpack(while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_28/unstack�
'while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'while/gru_cell_28/MatMul/ReadVariableOp�
while/gru_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul�
while/gru_cell_28/BiasAddBiasAdd"while/gru_cell_28/MatMul:product:0"while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAddt
while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_28/Const�
!while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!while/gru_cell_28/split/split_dim�
while/gru_cell_28/splitSplit*while/gru_cell_28/split/split_dim:output:0"while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split�
)while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02+
)while/gru_cell_28/MatMul_1/ReadVariableOp�
while/gru_cell_28/MatMul_1MatMulwhile_placeholder_21while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul_1�
while/gru_cell_28/BiasAdd_1BiasAdd$while/gru_cell_28/MatMul_1:product:0"while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAdd_1�
while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_28/Const_1�
#while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#while/gru_cell_28/split_1/split_dim�
while/gru_cell_28/split_1SplitV$while/gru_cell_28/BiasAdd_1:output:0"while/gru_cell_28/Const_1:output:0,while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split_1�
while/gru_cell_28/addAddV2 while/gru_cell_28/split:output:0"while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add�
while/gru_cell_28/SigmoidSigmoidwhile/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid�
while/gru_cell_28/add_1AddV2 while/gru_cell_28/split:output:1"while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_1�
while/gru_cell_28/Sigmoid_1Sigmoidwhile/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid_1�
while/gru_cell_28/mulMulwhile/gru_cell_28/Sigmoid_1:y:0"while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul�
while/gru_cell_28/add_2AddV2 while/gru_cell_28/split:output:2while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_2�
while/gru_cell_28/ReluReluwhile/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Relu�
while/gru_cell_28/mul_1Mulwhile/gru_cell_28/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_1w
while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_28/sub/x�
while/gru_cell_28/subSub while/gru_cell_28/sub/x:output:0while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/sub�
while/gru_cell_28/mul_2Mulwhile/gru_cell_28/sub:z:0$while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_2�
while/gru_cell_28/add_3AddV2while/gru_cell_28/mul_1:z:0while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_3�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_28/add_3:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"j
2while_gru_cell_28_matmul_1_readvariableop_resource4while_gru_cell_28_matmul_1_readvariableop_resource_0"f
0while_gru_cell_28_matmul_readvariableop_resource2while_gru_cell_28_matmul_readvariableop_resource_0"X
)while_gru_cell_28_readvariableop_resource+while_gru_cell_28_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2R
'while/gru_cell_28/MatMul/ReadVariableOp'while/gru_cell_28/MatMul/ReadVariableOp2V
)while/gru_cell_28/MatMul_1/ReadVariableOp)while/gru_cell_28/MatMul_1/ReadVariableOp2D
 while/gru_cell_28/ReadVariableOp while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�G
�
while_body_32655226
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_28_readvariableop_resource_06
2while_gru_cell_28_matmul_readvariableop_resource_08
4while_gru_cell_28_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_28_readvariableop_resource4
0while_gru_cell_28_matmul_readvariableop_resource6
2while_gru_cell_28_matmul_1_readvariableop_resource��'while/gru_cell_28/MatMul/ReadVariableOp�)while/gru_cell_28/MatMul_1/ReadVariableOp� while/gru_cell_28/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
 while/gru_cell_28/ReadVariableOpReadVariableOp+while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02"
 while/gru_cell_28/ReadVariableOp�
while/gru_cell_28/unstackUnpack(while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_28/unstack�
'while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'while/gru_cell_28/MatMul/ReadVariableOp�
while/gru_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul�
while/gru_cell_28/BiasAddBiasAdd"while/gru_cell_28/MatMul:product:0"while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAddt
while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_28/Const�
!while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!while/gru_cell_28/split/split_dim�
while/gru_cell_28/splitSplit*while/gru_cell_28/split/split_dim:output:0"while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split�
)while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02+
)while/gru_cell_28/MatMul_1/ReadVariableOp�
while/gru_cell_28/MatMul_1MatMulwhile_placeholder_21while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul_1�
while/gru_cell_28/BiasAdd_1BiasAdd$while/gru_cell_28/MatMul_1:product:0"while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAdd_1�
while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_28/Const_1�
#while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#while/gru_cell_28/split_1/split_dim�
while/gru_cell_28/split_1SplitV$while/gru_cell_28/BiasAdd_1:output:0"while/gru_cell_28/Const_1:output:0,while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split_1�
while/gru_cell_28/addAddV2 while/gru_cell_28/split:output:0"while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add�
while/gru_cell_28/SigmoidSigmoidwhile/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid�
while/gru_cell_28/add_1AddV2 while/gru_cell_28/split:output:1"while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_1�
while/gru_cell_28/Sigmoid_1Sigmoidwhile/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid_1�
while/gru_cell_28/mulMulwhile/gru_cell_28/Sigmoid_1:y:0"while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul�
while/gru_cell_28/add_2AddV2 while/gru_cell_28/split:output:2while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_2�
while/gru_cell_28/ReluReluwhile/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Relu�
while/gru_cell_28/mul_1Mulwhile/gru_cell_28/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_1w
while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_28/sub/x�
while/gru_cell_28/subSub while/gru_cell_28/sub/x:output:0while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/sub�
while/gru_cell_28/mul_2Mulwhile/gru_cell_28/sub:z:0$while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_2�
while/gru_cell_28/add_3AddV2while/gru_cell_28/mul_1:z:0while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_3�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_28/add_3:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"j
2while_gru_cell_28_matmul_1_readvariableop_resource4while_gru_cell_28_matmul_1_readvariableop_resource_0"f
0while_gru_cell_28_matmul_readvariableop_resource2while_gru_cell_28_matmul_readvariableop_resource_0"X
)while_gru_cell_28_readvariableop_resource+while_gru_cell_28_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2R
'while/gru_cell_28/MatMul/ReadVariableOp'while/gru_cell_28/MatMul/ReadVariableOp2V
)while/gru_cell_28/MatMul_1/ReadVariableOp)while/gru_cell_28/MatMul_1/ReadVariableOp2D
 while/gru_cell_28/ReadVariableOp while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�G
�
while_body_32656141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_28_readvariableop_resource_06
2while_gru_cell_28_matmul_readvariableop_resource_08
4while_gru_cell_28_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_28_readvariableop_resource4
0while_gru_cell_28_matmul_readvariableop_resource6
2while_gru_cell_28_matmul_1_readvariableop_resource��'while/gru_cell_28/MatMul/ReadVariableOp�)while/gru_cell_28/MatMul_1/ReadVariableOp� while/gru_cell_28/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
 while/gru_cell_28/ReadVariableOpReadVariableOp+while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02"
 while/gru_cell_28/ReadVariableOp�
while/gru_cell_28/unstackUnpack(while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_28/unstack�
'while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'while/gru_cell_28/MatMul/ReadVariableOp�
while/gru_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul�
while/gru_cell_28/BiasAddBiasAdd"while/gru_cell_28/MatMul:product:0"while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAddt
while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_28/Const�
!while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!while/gru_cell_28/split/split_dim�
while/gru_cell_28/splitSplit*while/gru_cell_28/split/split_dim:output:0"while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split�
)while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02+
)while/gru_cell_28/MatMul_1/ReadVariableOp�
while/gru_cell_28/MatMul_1MatMulwhile_placeholder_21while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul_1�
while/gru_cell_28/BiasAdd_1BiasAdd$while/gru_cell_28/MatMul_1:product:0"while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAdd_1�
while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_28/Const_1�
#while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#while/gru_cell_28/split_1/split_dim�
while/gru_cell_28/split_1SplitV$while/gru_cell_28/BiasAdd_1:output:0"while/gru_cell_28/Const_1:output:0,while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split_1�
while/gru_cell_28/addAddV2 while/gru_cell_28/split:output:0"while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add�
while/gru_cell_28/SigmoidSigmoidwhile/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid�
while/gru_cell_28/add_1AddV2 while/gru_cell_28/split:output:1"while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_1�
while/gru_cell_28/Sigmoid_1Sigmoidwhile/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid_1�
while/gru_cell_28/mulMulwhile/gru_cell_28/Sigmoid_1:y:0"while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul�
while/gru_cell_28/add_2AddV2 while/gru_cell_28/split:output:2while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_2�
while/gru_cell_28/ReluReluwhile/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Relu�
while/gru_cell_28/mul_1Mulwhile/gru_cell_28/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_1w
while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_28/sub/x�
while/gru_cell_28/subSub while/gru_cell_28/sub/x:output:0while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/sub�
while/gru_cell_28/mul_2Mulwhile/gru_cell_28/sub:z:0$while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_2�
while/gru_cell_28/add_3AddV2while/gru_cell_28/mul_1:z:0while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_3�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_28/add_3:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"j
2while_gru_cell_28_matmul_1_readvariableop_resource4while_gru_cell_28_matmul_1_readvariableop_resource_0"f
0while_gru_cell_28_matmul_readvariableop_resource2while_gru_cell_28_matmul_readvariableop_resource_0"X
)while_gru_cell_28_readvariableop_resource+while_gru_cell_28_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2R
'while/gru_cell_28/MatMul/ReadVariableOp'while/gru_cell_28/MatMul/ReadVariableOp2V
)while/gru_cell_28/MatMul_1/ReadVariableOp)while/gru_cell_28/MatMul_1/ReadVariableOp2D
 while/gru_cell_28/ReadVariableOp while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_78_layer_call_fn_32655621
gru_28_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_326556042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_namegru_28_input
�[
�
D__inference_gru_28_layer_call_and_return_conditional_losses_32655316

inputs'
#gru_cell_28_readvariableop_resource.
*gru_cell_28_matmul_readvariableop_resource0
,gru_cell_28_matmul_1_readvariableop_resource
identity��!gru_cell_28/MatMul/ReadVariableOp�#gru_cell_28/MatMul_1/ReadVariableOp�gru_cell_28/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_28/ReadVariableOpReadVariableOp#gru_cell_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_28/ReadVariableOp�
gru_cell_28/unstackUnpack"gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_28/unstack�
!gru_cell_28/MatMul/ReadVariableOpReadVariableOp*gru_cell_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!gru_cell_28/MatMul/ReadVariableOp�
gru_cell_28/MatMulMatMulstrided_slice_2:output:0)gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul�
gru_cell_28/BiasAddBiasAddgru_cell_28/MatMul:product:0gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAddh
gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_28/Const�
gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split/split_dim�
gru_cell_28/splitSplit$gru_cell_28/split/split_dim:output:0gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split�
#gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_28_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02%
#gru_cell_28/MatMul_1/ReadVariableOp�
gru_cell_28/MatMul_1MatMulzeros:output:0+gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_28/MatMul_1�
gru_cell_28/BiasAdd_1BiasAddgru_cell_28/MatMul_1:product:0gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_28/BiasAdd_1
gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_28/Const_1�
gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_28/split_1/split_dim�
gru_cell_28/split_1SplitVgru_cell_28/BiasAdd_1:output:0gru_cell_28/Const_1:output:0&gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_28/split_1�
gru_cell_28/addAddV2gru_cell_28/split:output:0gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add|
gru_cell_28/SigmoidSigmoidgru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid�
gru_cell_28/add_1AddV2gru_cell_28/split:output:1gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_1�
gru_cell_28/Sigmoid_1Sigmoidgru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Sigmoid_1�
gru_cell_28/mulMulgru_cell_28/Sigmoid_1:y:0gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul�
gru_cell_28/add_2AddV2gru_cell_28/split:output:2gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_2u
gru_cell_28/ReluRelugru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/Relu�
gru_cell_28/mul_1Mulgru_cell_28/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_1k
gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_28/sub/x�
gru_cell_28/subSubgru_cell_28/sub/x:output:0gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/sub�
gru_cell_28/mul_2Mulgru_cell_28/sub:z:0gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/mul_2�
gru_cell_28/add_3AddV2gru_cell_28/mul_1:z:0gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_28/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_28_readvariableop_resource*gru_cell_28_matmul_readvariableop_resource,gru_cell_28_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_32655226*
condR
while_cond_32655225*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0"^gru_cell_28/MatMul/ReadVariableOp$^gru_cell_28/MatMul_1/ReadVariableOp^gru_cell_28/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2F
!gru_cell_28/MatMul/ReadVariableOp!gru_cell_28/MatMul/ReadVariableOp2J
#gru_cell_28/MatMul_1/ReadVariableOp#gru_cell_28/MatMul_1/ReadVariableOp28
gru_cell_28/ReadVariableOpgru_cell_28/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
G__inference_dense_156_layer_call_and_return_conditional_losses_32655516

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������K::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������K
 
_user_specified_nameinputs
�i
�
(sequential_78_gru_28_while_body_32654489F
Bsequential_78_gru_28_while_sequential_78_gru_28_while_loop_counterL
Hsequential_78_gru_28_while_sequential_78_gru_28_while_maximum_iterations*
&sequential_78_gru_28_while_placeholder,
(sequential_78_gru_28_while_placeholder_1,
(sequential_78_gru_28_while_placeholder_2E
Asequential_78_gru_28_while_sequential_78_gru_28_strided_slice_1_0�
}sequential_78_gru_28_while_tensorarrayv2read_tensorlistgetitem_sequential_78_gru_28_tensorarrayunstack_tensorlistfromtensor_0D
@sequential_78_gru_28_while_gru_cell_28_readvariableop_resource_0K
Gsequential_78_gru_28_while_gru_cell_28_matmul_readvariableop_resource_0M
Isequential_78_gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0'
#sequential_78_gru_28_while_identity)
%sequential_78_gru_28_while_identity_1)
%sequential_78_gru_28_while_identity_2)
%sequential_78_gru_28_while_identity_3)
%sequential_78_gru_28_while_identity_4C
?sequential_78_gru_28_while_sequential_78_gru_28_strided_slice_1
{sequential_78_gru_28_while_tensorarrayv2read_tensorlistgetitem_sequential_78_gru_28_tensorarrayunstack_tensorlistfromtensorB
>sequential_78_gru_28_while_gru_cell_28_readvariableop_resourceI
Esequential_78_gru_28_while_gru_cell_28_matmul_readvariableop_resourceK
Gsequential_78_gru_28_while_gru_cell_28_matmul_1_readvariableop_resource��<sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp�>sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp�5sequential_78/gru_28/while/gru_cell_28/ReadVariableOp�
Lsequential_78/gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2N
Lsequential_78/gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shape�
>sequential_78/gru_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_78_gru_28_while_tensorarrayv2read_tensorlistgetitem_sequential_78_gru_28_tensorarrayunstack_tensorlistfromtensor_0&sequential_78_gru_28_while_placeholderUsequential_78/gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02@
>sequential_78/gru_28/while/TensorArrayV2Read/TensorListGetItem�
5sequential_78/gru_28/while/gru_cell_28/ReadVariableOpReadVariableOp@sequential_78_gru_28_while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype027
5sequential_78/gru_28/while/gru_cell_28/ReadVariableOp�
.sequential_78/gru_28/while/gru_cell_28/unstackUnpack=sequential_78/gru_28/while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num20
.sequential_78/gru_28/while/gru_cell_28/unstack�
<sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOpReadVariableOpGsequential_78_gru_28_while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02>
<sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp�
-sequential_78/gru_28/while/gru_cell_28/MatMulMatMulEsequential_78/gru_28/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2/
-sequential_78/gru_28/while/gru_cell_28/MatMul�
.sequential_78/gru_28/while/gru_cell_28/BiasAddBiasAdd7sequential_78/gru_28/while/gru_cell_28/MatMul:product:07sequential_78/gru_28/while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������20
.sequential_78/gru_28/while/gru_cell_28/BiasAdd�
,sequential_78/gru_28/while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_78/gru_28/while/gru_cell_28/Const�
6sequential_78/gru_28/while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������28
6sequential_78/gru_28/while/gru_cell_28/split/split_dim�
,sequential_78/gru_28/while/gru_cell_28/splitSplit?sequential_78/gru_28/while/gru_cell_28/split/split_dim:output:07sequential_78/gru_28/while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2.
,sequential_78/gru_28/while/gru_cell_28/split�
>sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOpIsequential_78_gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02@
>sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp�
/sequential_78/gru_28/while/gru_cell_28/MatMul_1MatMul(sequential_78_gru_28_while_placeholder_2Fsequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_78/gru_28/while/gru_cell_28/MatMul_1�
0sequential_78/gru_28/while/gru_cell_28/BiasAdd_1BiasAdd9sequential_78/gru_28/while/gru_cell_28/MatMul_1:product:07sequential_78/gru_28/while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������22
0sequential_78/gru_28/while/gru_cell_28/BiasAdd_1�
.sequential_78/gru_28/while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����20
.sequential_78/gru_28/while/gru_cell_28/Const_1�
8sequential_78/gru_28/while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8sequential_78/gru_28/while/gru_cell_28/split_1/split_dim�
.sequential_78/gru_28/while/gru_cell_28/split_1SplitV9sequential_78/gru_28/while/gru_cell_28/BiasAdd_1:output:07sequential_78/gru_28/while/gru_cell_28/Const_1:output:0Asequential_78/gru_28/while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split20
.sequential_78/gru_28/while/gru_cell_28/split_1�
*sequential_78/gru_28/while/gru_cell_28/addAddV25sequential_78/gru_28/while/gru_cell_28/split:output:07sequential_78/gru_28/while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2,
*sequential_78/gru_28/while/gru_cell_28/add�
.sequential_78/gru_28/while/gru_cell_28/SigmoidSigmoid.sequential_78/gru_28/while/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K20
.sequential_78/gru_28/while/gru_cell_28/Sigmoid�
,sequential_78/gru_28/while/gru_cell_28/add_1AddV25sequential_78/gru_28/while/gru_cell_28/split:output:17sequential_78/gru_28/while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2.
,sequential_78/gru_28/while/gru_cell_28/add_1�
0sequential_78/gru_28/while/gru_cell_28/Sigmoid_1Sigmoid0sequential_78/gru_28/while/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K22
0sequential_78/gru_28/while/gru_cell_28/Sigmoid_1�
*sequential_78/gru_28/while/gru_cell_28/mulMul4sequential_78/gru_28/while/gru_cell_28/Sigmoid_1:y:07sequential_78/gru_28/while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2,
*sequential_78/gru_28/while/gru_cell_28/mul�
,sequential_78/gru_28/while/gru_cell_28/add_2AddV25sequential_78/gru_28/while/gru_cell_28/split:output:2.sequential_78/gru_28/while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2.
,sequential_78/gru_28/while/gru_cell_28/add_2�
+sequential_78/gru_28/while/gru_cell_28/ReluRelu0sequential_78/gru_28/while/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2-
+sequential_78/gru_28/while/gru_cell_28/Relu�
,sequential_78/gru_28/while/gru_cell_28/mul_1Mul2sequential_78/gru_28/while/gru_cell_28/Sigmoid:y:0(sequential_78_gru_28_while_placeholder_2*
T0*'
_output_shapes
:���������K2.
,sequential_78/gru_28/while/gru_cell_28/mul_1�
,sequential_78/gru_28/while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,sequential_78/gru_28/while/gru_cell_28/sub/x�
*sequential_78/gru_28/while/gru_cell_28/subSub5sequential_78/gru_28/while/gru_cell_28/sub/x:output:02sequential_78/gru_28/while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2,
*sequential_78/gru_28/while/gru_cell_28/sub�
,sequential_78/gru_28/while/gru_cell_28/mul_2Mul.sequential_78/gru_28/while/gru_cell_28/sub:z:09sequential_78/gru_28/while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2.
,sequential_78/gru_28/while/gru_cell_28/mul_2�
,sequential_78/gru_28/while/gru_cell_28/add_3AddV20sequential_78/gru_28/while/gru_cell_28/mul_1:z:00sequential_78/gru_28/while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2.
,sequential_78/gru_28/while/gru_cell_28/add_3�
?sequential_78/gru_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_78_gru_28_while_placeholder_1&sequential_78_gru_28_while_placeholder0sequential_78/gru_28/while/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_78/gru_28/while/TensorArrayV2Write/TensorListSetItem�
 sequential_78/gru_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_78/gru_28/while/add/y�
sequential_78/gru_28/while/addAddV2&sequential_78_gru_28_while_placeholder)sequential_78/gru_28/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_78/gru_28/while/add�
"sequential_78/gru_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_78/gru_28/while/add_1/y�
 sequential_78/gru_28/while/add_1AddV2Bsequential_78_gru_28_while_sequential_78_gru_28_while_loop_counter+sequential_78/gru_28/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_78/gru_28/while/add_1�
#sequential_78/gru_28/while/IdentityIdentity$sequential_78/gru_28/while/add_1:z:0=^sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp?^sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp6^sequential_78/gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_78/gru_28/while/Identity�
%sequential_78/gru_28/while/Identity_1IdentityHsequential_78_gru_28_while_sequential_78_gru_28_while_maximum_iterations=^sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp?^sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp6^sequential_78/gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_78/gru_28/while/Identity_1�
%sequential_78/gru_28/while/Identity_2Identity"sequential_78/gru_28/while/add:z:0=^sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp?^sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp6^sequential_78/gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_78/gru_28/while/Identity_2�
%sequential_78/gru_28/while/Identity_3IdentityOsequential_78/gru_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp?^sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp6^sequential_78/gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_78/gru_28/while/Identity_3�
%sequential_78/gru_28/while/Identity_4Identity0sequential_78/gru_28/while/gru_cell_28/add_3:z:0=^sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp?^sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp6^sequential_78/gru_28/while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2'
%sequential_78/gru_28/while/Identity_4"�
Gsequential_78_gru_28_while_gru_cell_28_matmul_1_readvariableop_resourceIsequential_78_gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0"�
Esequential_78_gru_28_while_gru_cell_28_matmul_readvariableop_resourceGsequential_78_gru_28_while_gru_cell_28_matmul_readvariableop_resource_0"�
>sequential_78_gru_28_while_gru_cell_28_readvariableop_resource@sequential_78_gru_28_while_gru_cell_28_readvariableop_resource_0"S
#sequential_78_gru_28_while_identity,sequential_78/gru_28/while/Identity:output:0"W
%sequential_78_gru_28_while_identity_1.sequential_78/gru_28/while/Identity_1:output:0"W
%sequential_78_gru_28_while_identity_2.sequential_78/gru_28/while/Identity_2:output:0"W
%sequential_78_gru_28_while_identity_3.sequential_78/gru_28/while/Identity_3:output:0"W
%sequential_78_gru_28_while_identity_4.sequential_78/gru_28/while/Identity_4:output:0"�
?sequential_78_gru_28_while_sequential_78_gru_28_strided_slice_1Asequential_78_gru_28_while_sequential_78_gru_28_strided_slice_1_0"�
{sequential_78_gru_28_while_tensorarrayv2read_tensorlistgetitem_sequential_78_gru_28_tensorarrayunstack_tensorlistfromtensor}sequential_78_gru_28_while_tensorarrayv2read_tensorlistgetitem_sequential_78_gru_28_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2|
<sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp<sequential_78/gru_28/while/gru_cell_28/MatMul/ReadVariableOp2�
>sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp>sequential_78/gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp2n
5sequential_78/gru_28/while/gru_cell_28/ReadVariableOp5sequential_78/gru_28/while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32654664

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������K
 
_user_specified_namestates
�	
�
gru_28_while_cond_32655758*
&gru_28_while_gru_28_while_loop_counter0
,gru_28_while_gru_28_while_maximum_iterations
gru_28_while_placeholder
gru_28_while_placeholder_1
gru_28_while_placeholder_2,
(gru_28_while_less_gru_28_strided_slice_1D
@gru_28_while_gru_28_while_cond_32655758___redundant_placeholder0D
@gru_28_while_gru_28_while_cond_32655758___redundant_placeholder1D
@gru_28_while_gru_28_while_cond_32655758___redundant_placeholder2D
@gru_28_while_gru_28_while_cond_32655758___redundant_placeholder3
gru_28_while_identity
�
gru_28/while/LessLessgru_28_while_placeholder(gru_28_while_less_gru_28_strided_slice_1*
T0*
_output_shapes
: 2
gru_28/while/Lessr
gru_28/while/IdentityIdentitygru_28/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_28/while/Identity"7
gru_28_while_identitygru_28/while/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_32656639
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_32656639___redundant_placeholder06
2while_while_cond_32656639___redundant_placeholder16
2while_while_cond_32656639___redundant_placeholder26
2while_while_cond_32656639___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
:
�G
�
while_body_32656300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_28_readvariableop_resource_06
2while_gru_cell_28_matmul_readvariableop_resource_08
4while_gru_cell_28_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_28_readvariableop_resource4
0while_gru_cell_28_matmul_readvariableop_resource6
2while_gru_cell_28_matmul_1_readvariableop_resource��'while/gru_cell_28/MatMul/ReadVariableOp�)while/gru_cell_28/MatMul_1/ReadVariableOp� while/gru_cell_28/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
 while/gru_cell_28/ReadVariableOpReadVariableOp+while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02"
 while/gru_cell_28/ReadVariableOp�
while/gru_cell_28/unstackUnpack(while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_28/unstack�
'while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'while/gru_cell_28/MatMul/ReadVariableOp�
while/gru_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul�
while/gru_cell_28/BiasAddBiasAdd"while/gru_cell_28/MatMul:product:0"while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAddt
while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_28/Const�
!while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!while/gru_cell_28/split/split_dim�
while/gru_cell_28/splitSplit*while/gru_cell_28/split/split_dim:output:0"while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split�
)while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02+
)while/gru_cell_28/MatMul_1/ReadVariableOp�
while/gru_cell_28/MatMul_1MatMulwhile_placeholder_21while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_28/MatMul_1�
while/gru_cell_28/BiasAdd_1BiasAdd$while/gru_cell_28/MatMul_1:product:0"while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_28/BiasAdd_1�
while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_28/Const_1�
#while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#while/gru_cell_28/split_1/split_dim�
while/gru_cell_28/split_1SplitV$while/gru_cell_28/BiasAdd_1:output:0"while/gru_cell_28/Const_1:output:0,while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_28/split_1�
while/gru_cell_28/addAddV2 while/gru_cell_28/split:output:0"while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add�
while/gru_cell_28/SigmoidSigmoidwhile/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid�
while/gru_cell_28/add_1AddV2 while/gru_cell_28/split:output:1"while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_1�
while/gru_cell_28/Sigmoid_1Sigmoidwhile/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Sigmoid_1�
while/gru_cell_28/mulMulwhile/gru_cell_28/Sigmoid_1:y:0"while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul�
while/gru_cell_28/add_2AddV2 while/gru_cell_28/split:output:2while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_2�
while/gru_cell_28/ReluReluwhile/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/Relu�
while/gru_cell_28/mul_1Mulwhile/gru_cell_28/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_1w
while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_28/sub/x�
while/gru_cell_28/subSub while/gru_cell_28/sub/x:output:0while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/sub�
while/gru_cell_28/mul_2Mulwhile/gru_cell_28/sub:z:0$while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/mul_2�
while/gru_cell_28/add_3AddV2while/gru_cell_28/mul_1:z:0while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_28/add_3�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_28/add_3:z:0(^while/gru_cell_28/MatMul/ReadVariableOp*^while/gru_cell_28/MatMul_1/ReadVariableOp!^while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"j
2while_gru_cell_28_matmul_1_readvariableop_resource4while_gru_cell_28_matmul_1_readvariableop_resource_0"f
0while_gru_cell_28_matmul_readvariableop_resource2while_gru_cell_28_matmul_readvariableop_resource_0"X
)while_gru_cell_28_readvariableop_resource+while_gru_cell_28_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2R
'while/gru_cell_28/MatMul/ReadVariableOp'while/gru_cell_28/MatMul/ReadVariableOp2V
)while/gru_cell_28/MatMul_1/ReadVariableOp)while/gru_cell_28/MatMul_1/ReadVariableOp2D
 while/gru_cell_28/ReadVariableOp while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_78_layer_call_fn_32656053

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_sequential_78_layer_call_and_return_conditional_losses_326556042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�	
gru_28_while_body_32655759*
&gru_28_while_gru_28_while_loop_counter0
,gru_28_while_gru_28_while_maximum_iterations
gru_28_while_placeholder
gru_28_while_placeholder_1
gru_28_while_placeholder_2)
%gru_28_while_gru_28_strided_slice_1_0e
agru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_06
2gru_28_while_gru_cell_28_readvariableop_resource_0=
9gru_28_while_gru_cell_28_matmul_readvariableop_resource_0?
;gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0
gru_28_while_identity
gru_28_while_identity_1
gru_28_while_identity_2
gru_28_while_identity_3
gru_28_while_identity_4'
#gru_28_while_gru_28_strided_slice_1c
_gru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor4
0gru_28_while_gru_cell_28_readvariableop_resource;
7gru_28_while_gru_cell_28_matmul_readvariableop_resource=
9gru_28_while_gru_cell_28_matmul_1_readvariableop_resource��.gru_28/while/gru_cell_28/MatMul/ReadVariableOp�0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp�'gru_28/while/gru_cell_28/ReadVariableOp�
>gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2@
>gru_28/while/TensorArrayV2Read/TensorListGetItem/element_shape�
0gru_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0gru_28_while_placeholderGgru_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype022
0gru_28/while/TensorArrayV2Read/TensorListGetItem�
'gru_28/while/gru_cell_28/ReadVariableOpReadVariableOp2gru_28_while_gru_cell_28_readvariableop_resource_0*
_output_shapes
:	�*
dtype02)
'gru_28/while/gru_cell_28/ReadVariableOp�
 gru_28/while/gru_cell_28/unstackUnpack/gru_28/while/gru_cell_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2"
 gru_28/while/gru_cell_28/unstack�
.gru_28/while/gru_cell_28/MatMul/ReadVariableOpReadVariableOp9gru_28_while_gru_cell_28_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype020
.gru_28/while/gru_cell_28/MatMul/ReadVariableOp�
gru_28/while/gru_cell_28/MatMulMatMul7gru_28/while/TensorArrayV2Read/TensorListGetItem:item:06gru_28/while/gru_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
gru_28/while/gru_cell_28/MatMul�
 gru_28/while/gru_cell_28/BiasAddBiasAdd)gru_28/while/gru_cell_28/MatMul:product:0)gru_28/while/gru_cell_28/unstack:output:0*
T0*(
_output_shapes
:����������2"
 gru_28/while/gru_cell_28/BiasAdd�
gru_28/while/gru_cell_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_28/while/gru_cell_28/Const�
(gru_28/while/gru_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(gru_28/while/gru_cell_28/split/split_dim�
gru_28/while/gru_cell_28/splitSplit1gru_28/while/gru_cell_28/split/split_dim:output:0)gru_28/while/gru_cell_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2 
gru_28/while/gru_cell_28/split�
0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOpReadVariableOp;gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype022
0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp�
!gru_28/while/gru_cell_28/MatMul_1MatMulgru_28_while_placeholder_28gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!gru_28/while/gru_cell_28/MatMul_1�
"gru_28/while/gru_cell_28/BiasAdd_1BiasAdd+gru_28/while/gru_cell_28/MatMul_1:product:0)gru_28/while/gru_cell_28/unstack:output:1*
T0*(
_output_shapes
:����������2$
"gru_28/while/gru_cell_28/BiasAdd_1�
 gru_28/while/gru_cell_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2"
 gru_28/while/gru_cell_28/Const_1�
*gru_28/while/gru_cell_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*gru_28/while/gru_cell_28/split_1/split_dim�
 gru_28/while/gru_cell_28/split_1SplitV+gru_28/while/gru_cell_28/BiasAdd_1:output:0)gru_28/while/gru_cell_28/Const_1:output:03gru_28/while/gru_cell_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2"
 gru_28/while/gru_cell_28/split_1�
gru_28/while/gru_cell_28/addAddV2'gru_28/while/gru_cell_28/split:output:0)gru_28/while/gru_cell_28/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/add�
 gru_28/while/gru_cell_28/SigmoidSigmoid gru_28/while/gru_cell_28/add:z:0*
T0*'
_output_shapes
:���������K2"
 gru_28/while/gru_cell_28/Sigmoid�
gru_28/while/gru_cell_28/add_1AddV2'gru_28/while/gru_cell_28/split:output:1)gru_28/while/gru_cell_28/split_1:output:1*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/add_1�
"gru_28/while/gru_cell_28/Sigmoid_1Sigmoid"gru_28/while/gru_cell_28/add_1:z:0*
T0*'
_output_shapes
:���������K2$
"gru_28/while/gru_cell_28/Sigmoid_1�
gru_28/while/gru_cell_28/mulMul&gru_28/while/gru_cell_28/Sigmoid_1:y:0)gru_28/while/gru_cell_28/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/mul�
gru_28/while/gru_cell_28/add_2AddV2'gru_28/while/gru_cell_28/split:output:2 gru_28/while/gru_cell_28/mul:z:0*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/add_2�
gru_28/while/gru_cell_28/ReluRelu"gru_28/while/gru_cell_28/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/Relu�
gru_28/while/gru_cell_28/mul_1Mul$gru_28/while/gru_cell_28/Sigmoid:y:0gru_28_while_placeholder_2*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/mul_1�
gru_28/while/gru_cell_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
gru_28/while/gru_cell_28/sub/x�
gru_28/while/gru_cell_28/subSub'gru_28/while/gru_cell_28/sub/x:output:0$gru_28/while/gru_cell_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_28/while/gru_cell_28/sub�
gru_28/while/gru_cell_28/mul_2Mul gru_28/while/gru_cell_28/sub:z:0+gru_28/while/gru_cell_28/Relu:activations:0*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/mul_2�
gru_28/while/gru_cell_28/add_3AddV2"gru_28/while/gru_cell_28/mul_1:z:0"gru_28/while/gru_cell_28/mul_2:z:0*
T0*'
_output_shapes
:���������K2 
gru_28/while/gru_cell_28/add_3�
1gru_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_28_while_placeholder_1gru_28_while_placeholder"gru_28/while/gru_cell_28/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_28/while/TensorArrayV2Write/TensorListSetItemj
gru_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/while/add/y�
gru_28/while/addAddV2gru_28_while_placeholdergru_28/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_28/while/addn
gru_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/while/add_1/y�
gru_28/while/add_1AddV2&gru_28_while_gru_28_while_loop_countergru_28/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_28/while/add_1�
gru_28/while/IdentityIdentitygru_28/while/add_1:z:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity�
gru_28/while/Identity_1Identity,gru_28_while_gru_28_while_maximum_iterations/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity_1�
gru_28/while/Identity_2Identitygru_28/while/add:z:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity_2�
gru_28/while/Identity_3IdentityAgru_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*
_output_shapes
: 2
gru_28/while/Identity_3�
gru_28/while/Identity_4Identity"gru_28/while/gru_cell_28/add_3:z:0/^gru_28/while/gru_cell_28/MatMul/ReadVariableOp1^gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp(^gru_28/while/gru_cell_28/ReadVariableOp*
T0*'
_output_shapes
:���������K2
gru_28/while/Identity_4"L
#gru_28_while_gru_28_strided_slice_1%gru_28_while_gru_28_strided_slice_1_0"x
9gru_28_while_gru_cell_28_matmul_1_readvariableop_resource;gru_28_while_gru_cell_28_matmul_1_readvariableop_resource_0"t
7gru_28_while_gru_cell_28_matmul_readvariableop_resource9gru_28_while_gru_cell_28_matmul_readvariableop_resource_0"f
0gru_28_while_gru_cell_28_readvariableop_resource2gru_28_while_gru_cell_28_readvariableop_resource_0"7
gru_28_while_identitygru_28/while/Identity:output:0";
gru_28_while_identity_1 gru_28/while/Identity_1:output:0";
gru_28_while_identity_2 gru_28/while/Identity_2:output:0";
gru_28_while_identity_3 gru_28/while/Identity_3:output:0";
gru_28_while_identity_4 gru_28/while/Identity_4:output:0"�
_gru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensoragru_28_while_tensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2`
.gru_28/while/gru_cell_28/MatMul/ReadVariableOp.gru_28/while/gru_cell_28/MatMul/ReadVariableOp2d
0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp0gru_28/while/gru_cell_28/MatMul_1/ReadVariableOp2R
'gru_28/while/gru_cell_28/ReadVariableOp'gru_28/while/gru_cell_28/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32656871

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�	
�
G__inference_dense_157_layer_call_and_return_conditional_losses_32655542

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
gru_28_input9
serving_default_gru_28_input:0���������=
	dense_1570
StatefulPartitionedCall:0���������tensorflow/serving/predict:Ҿ
�*
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
*a&call_and_return_all_conditional_losses
b_default_save_signature
c__call__"�(
_tf_keras_sequential�({"class_name": "Sequential", "name": "sequential_78", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_78", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_28_input"}}, {"class_name": "GRU", "config": {"name": "gru_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_156", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_157", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_78", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_28_input"}}, {"class_name": "GRU", "config": {"name": "gru_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_156", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_157", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0010000000474974513, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"�
_tf_keras_rnn_layer�
{"class_name": "GRU", "name": "gru_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 1]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_156", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_156", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*h&call_and_return_all_conditional_losses
i__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_157", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_157", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
iter

beta_1

beta_2
	decay
 learning_ratemSmTmUmV!mW"mX#mYvZv[v\v]!v^"v_#v`"
	optimizer
Q
!0
"1
#2
3
4
5
6"
trackable_list_wrapper
Q
!0
"1
#2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
$layer_metrics
	variables
%metrics

&layers
'layer_regularization_losses
regularization_losses
(non_trainable_variables
c__call__
b_default_save_signature
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
�

!kernel
"recurrent_kernel
#bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
*k&call_and_return_all_conditional_losses
l__call__"�
_tf_keras_layer�{"class_name": "GRUCell", "name": "gru_cell_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_28", "trainable": true, "dtype": "float32", "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
-layer_metrics
	variables

.states
/metrics

0layers
1layer_regularization_losses
regularization_losses
2non_trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
": Kd2dense_156/kernel
:d2dense_156/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
3layer_metrics
	variables
4metrics

5layers
6layer_regularization_losses
regularization_losses
7non_trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
": d2dense_157/kernel
:2dense_157/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
8layer_metrics
	variables
9metrics

:layers
;layer_regularization_losses
regularization_losses
<non_trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	�2gru_28/gru_cell_28/kernel
6:4	K�2#gru_28/gru_cell_28/recurrent_kernel
*:(	�2gru_28/gru_cell_28/bias
 "
trackable_dict_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)trainable_variables
@layer_metrics
*	variables
Ametrics

Blayers
Clayer_regularization_losses
+regularization_losses
Dnon_trainable_variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Etotal
	Fcount
G	variables
H	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Itotal
	Jcount
K
_fn_kwargs
L	variables
M	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
�
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
E0
F1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
':%Kd2Adam/dense_156/kernel/m
!:d2Adam/dense_156/bias/m
':%d2Adam/dense_157/kernel/m
!:2Adam/dense_157/bias/m
1:/	�2 Adam/gru_28/gru_cell_28/kernel/m
;:9	K�2*Adam/gru_28/gru_cell_28/recurrent_kernel/m
/:-	�2Adam/gru_28/gru_cell_28/bias/m
':%Kd2Adam/dense_156/kernel/v
!:d2Adam/dense_156/bias/v
':%d2Adam/dense_157/kernel/v
!:2Adam/dense_157/bias/v
1:/	�2 Adam/gru_28/gru_cell_28/kernel/v
;:9	K�2*Adam/gru_28/gru_cell_28/recurrent_kernel/v
/:-	�2Adam/gru_28/gru_cell_28/bias/v
�2�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32656034
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655580
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655862
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655559�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_32654592�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� */�,
*�'
gru_28_input���������
�2�
0__inference_sequential_78_layer_call_fn_32655661
0__inference_sequential_78_layer_call_fn_32655621
0__inference_sequential_78_layer_call_fn_32656072
0__inference_sequential_78_layer_call_fn_32656053�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_gru_28_layer_call_and_return_conditional_losses_32656571
D__inference_gru_28_layer_call_and_return_conditional_losses_32656730
D__inference_gru_28_layer_call_and_return_conditional_losses_32656390
D__inference_gru_28_layer_call_and_return_conditional_losses_32656231�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_gru_28_layer_call_fn_32656412
)__inference_gru_28_layer_call_fn_32656752
)__inference_gru_28_layer_call_fn_32656401
)__inference_gru_28_layer_call_fn_32656741�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dense_156_layer_call_and_return_conditional_losses_32656763�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_dense_156_layer_call_fn_32656772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dense_157_layer_call_and_return_conditional_losses_32656782�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_dense_157_layer_call_fn_32656791�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_32655690gru_28_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32656871
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32656831�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_gru_cell_28_layer_call_fn_32656899
.__inference_gru_cell_28_layer_call_fn_32656885�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
#__inference__wrapped_model_32654592{#!"9�6
/�,
*�'
gru_28_input���������
� "5�2
0
	dense_157#� 
	dense_157����������
G__inference_dense_156_layer_call_and_return_conditional_losses_32656763\/�,
%�"
 �
inputs���������K
� "%�"
�
0���������d
� 
,__inference_dense_156_layer_call_fn_32656772O/�,
%�"
 �
inputs���������K
� "����������d�
G__inference_dense_157_layer_call_and_return_conditional_losses_32656782\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� 
,__inference_dense_157_layer_call_fn_32656791O/�,
%�"
 �
inputs���������d
� "�����������
D__inference_gru_28_layer_call_and_return_conditional_losses_32656231}#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������K
� �
D__inference_gru_28_layer_call_and_return_conditional_losses_32656390}#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������K
� �
D__inference_gru_28_layer_call_and_return_conditional_losses_32656571m#!"?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0���������K
� �
D__inference_gru_28_layer_call_and_return_conditional_losses_32656730m#!"?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0���������K
� �
)__inference_gru_28_layer_call_fn_32656401p#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������K�
)__inference_gru_28_layer_call_fn_32656412p#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������K�
)__inference_gru_28_layer_call_fn_32656741`#!"?�<
5�2
$�!
inputs���������

 
p

 
� "����������K�
)__inference_gru_28_layer_call_fn_32656752`#!"?�<
5�2
$�!
inputs���������

 
p 

 
� "����������K�
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32656831�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p
� "R�O
H�E
�
0/0���������K
$�!
�
0/1/0���������K
� �
I__inference_gru_cell_28_layer_call_and_return_conditional_losses_32656871�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p 
� "R�O
H�E
�
0/0���������K
$�!
�
0/1/0���������K
� �
.__inference_gru_cell_28_layer_call_fn_32656885�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p
� "D�A
�
0���������K
"�
�
1/0���������K�
.__inference_gru_cell_28_layer_call_fn_32656899�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p 
� "D�A
�
0���������K
"�
�
1/0���������K�
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655559s#!"A�>
7�4
*�'
gru_28_input���������
p

 
� "%�"
�
0���������
� �
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655580s#!"A�>
7�4
*�'
gru_28_input���������
p 

 
� "%�"
�
0���������
� �
K__inference_sequential_78_layer_call_and_return_conditional_losses_32655862m#!";�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
K__inference_sequential_78_layer_call_and_return_conditional_losses_32656034m#!";�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
0__inference_sequential_78_layer_call_fn_32655621f#!"A�>
7�4
*�'
gru_28_input���������
p

 
� "�����������
0__inference_sequential_78_layer_call_fn_32655661f#!"A�>
7�4
*�'
gru_28_input���������
p 

 
� "�����������
0__inference_sequential_78_layer_call_fn_32656053`#!";�8
1�.
$�!
inputs���������
p

 
� "�����������
0__inference_sequential_78_layer_call_fn_32656072`#!";�8
1�.
$�!
inputs���������
p 

 
� "�����������
&__inference_signature_wrapper_32655690�#!"I�F
� 
?�<
:
gru_28_input*�'
gru_28_input���������"5�2
0
	dense_157#� 
	dense_157���������