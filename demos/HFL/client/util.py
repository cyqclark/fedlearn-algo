# Copyright 2021 Fedlearn authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os,sys
from  abc import ABC, abstractmethod


root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from core.entity.common.machineinfo import MachineInfo
from demos.HFL.common.hfl_message import HFL_MSG
from demos.HFL.basic_control_msg_type import HFL_Control_Massage_Type as CMT
from demos.HFL.common.param_util import(
    Params, 
    ParamsRes, 
    TrainArgs, 
    TrainRes, 
    EvalArgs,
    TrainRes,
    NLPInferArgs,
    NLPInferRes
)


def params_to_msg(
                sender:MachineInfo, 
                receiver:MachineInfo,
                params:Params)->HFL_MSG:

        msg =  HFL_MSG('',sender, receiver)
        
        #msg.type = DMT.MSG_TRAIN_RES_C2S
        param_dict = {n:w for w,n in zip(params.weights, params.names)} 
        
        msg.add(
                HFL_MSG.KEY_PARAMS_WEIGHTS,
                param_dict
        )
             
        return msg

def msg_to_modelParam(msg:HFL_MSG):

        names,weights = zip(*[(k,v) 
            for k,v in msg.params[HFL_MSG.KEY_PARAMS_WEIGHTS].items()])

        params = Params(weights=weights, 
                        names=names, 
                        weight_type='float')
        return params 


def trainArgs_to_msg(
        sender:MachineInfo, 
        receiver:MachineInfo,
        trainArgs: TrainArgs)->HFL_MSG:
            
        msg = params_to_msg(sender, receiver, trainArgs.params)

        msg.add(
                HFL_MSG.KEY_CONFIG_TRAIN,
                trainArgs.config)
        msg.set_type(CMT.CTRL_TRAIN_S2C)        
        
        return msg


def trainRes_to_msg(
        sender:MachineInfo, 
        receiver:MachineInfo,
        trainRes: TrainRes)->HFL_MSG:
            
        msg = params_to_msg(sender, receiver, trainRes.params)

        msg.add(
            HFL_MSG.KEY_NUM_TRAIN_SAMPLES,
            trainRes.num_samples)

        msg.add(
                HFL_MSG.KEY_METRICS,
                trainRes.metrics)
        
        msg.set_type(CMT.MSG_TRAIN_RES_C2S)

        return msg        

def NLPInferArgs_to_msg(
        sender:MachineInfo, 
        receiver:MachineInfo,
        nlp_inferAgrs: NLPInferArgs)->HFL_MSG:

        if nlp_inferAgrs.params is None:
                msg =  HFL_MSG('',sender, receiver)
                msg.add(HFL_MSG.KEY_PARAMS_WEIGHTS,None)
        else: 
                msg = params_to_msg(sender, receiver, nlp_inferAgrs.params) 
        
        msg.set_type(CMT.CTRL_NLP_INFER_S2C)
        msg.add(HFL_MSG.KEY_INFER_SENTENCES,
            nlp_inferAgrs.inputs
        )
        return msg

def msg_to_NLPInferArgs(msg:HFL_MSG)->NLPInferArgs:
        params = None if msg.params[HFL_MSG.KEY_PARAMS_WEIGHTS] is None \
                else msg_to_modelParam(msg)
        nlpInferAgrs = NLPInferArgs(params=params,
                        inputs=msg[HFL_MSG.KEY_INFER_SENTENCES])
        return nlpInferAgrs

def NLPInferRes_to_msg(
        sender:MachineInfo, 
        receiver:MachineInfo,
        nlp_inferRes: NLPInferRes)->HFL_MSG:
    
    msg =  HFL_MSG(CMT.MSG_NLP_INFER_RES_C2S, sender, receiver)
    
    msg.params[HFL_MSG.KEY_INFER_SENTENCES] =   nlp_inferRes.inputs
    msg.params[HFL_MSG.KEY_INFER_SENTENCES_REP] =   nlp_inferRes.outputs

    return msg

def msg_to_NLPInferRes(msg:HFL_MSG)->NLPInferRes:
        return NLPInferRes(inputs = msg.params[HFL_MSG.KEY_INFER_SENTENCES],
                    outputs=msg.params[HFL_MSG.KEY_INFER_SENTENCES_REP])


def msg_to_trainArgs(msg:HFL_MSG):
        
        modelParam = msg_to_modelParam(msg)    

        trainArg = TrainArgs(params=modelParam, 
                            config=msg.params[HFL_MSG.KEY_CONFIG_TRAIN])    

        return trainArg

def msg_to_trainRes(msg:HFL_MSG):
    
    modelParam = msg_to_modelParam(msg)


    trainRes = TrainRes(params=modelParam,
                        num_samples = msg.params[HFL_MSG.KEY_NUM_TRAIN_SAMPLES], 
                        metrics = msg.params[HFL_MSG.KEY_METRICS])    

    return trainRes



   