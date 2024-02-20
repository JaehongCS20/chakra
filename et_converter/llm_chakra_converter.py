import logging
from io import TextIOWrapper
from typing import Optional, List
from third_party.utils.protolib import encodeMessage as encode_message
from et_def.et_def_pb2 import *

class Layer:
    def __init__(self, line: str):
        try:
            col = line.strip().split()
            self.name = col[0]

            # compuation
            self.comp_time = int(col[1])
            self.comp_node = None

            # memory
            self.input_memory_loc = str(col[2])
            self.input_memory_size = int(col[3])
            self.input_memory_node = None
            self.weight_memory_loc = str(col[4])
            self.weight_memory_size = int(col[5])
            self.weight_memory_node = None
            self.output_memory_loc = str(col[6])
            self.output_memory_size = int(col[7])
            self.output_memory_node = None

            # communication
            self.comm_type = str(col[8])
            self.comm_size = int(col[9])
            self.comm_node = None

            # Used for tensor & pipeline parallel
            self.comp_node_list = list()
            self.input_memory_node_list = list()
            self.weight_memory_node_list = list()
            self.output_memory_node_list = list()
            self.comm_node_list = list()

            self.misc = str(col[10])
        except:
            raise ValueError(f"Cannot parse the following layer -- \"{line}\"")
        
class LLMChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_dims: int,
        num_npus: int,
        logger: logging.Logger
    ):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_dims = num_dims
        self.num_npus = num_npus
        self.logger = logger
        self.next_node_id = 0

        # For send & recv nodes
        self.next_comm_tag = 0
        self.comm_tag_dict = dict()
    
    @staticmethod
    def get_layers(f: TextIOWrapper) -> List[Layer]:
        layers: List[Layer] = []
        for line in f:
            layers.append(Layer(line))
        return layers

    def get_next_node_id(self) -> int:
        ret = self.next_node_id
        self.next_node_id += 1
        return ret
    
    def get_next_comm_tag(self) -> int:
        ret = self.next_comm_tag
        self.next_comm_tag += 1
        return ret

    def get_node(self, name: str, node_type: int) -> Node:
        node = Node()
        node.id = self.get_next_node_id()
        node.name = name
        node.node_type = node_type
        return node

    def get_comp_node(self, layer_name: str, comp_time: int) -> Node:
        node = self.get_node("COMP_NODE_" + layer_name, COMP_NODE)
        node.simulated_run_time = comp_time
        return node

    def get_memory_load_node(self, layer_name: str, tensor_type: str, mem_type: str, mem_size: int) -> Node:
        node = self.get_node("MEM_LOAD_NODE_" + layer_name + "_" + tensor_type, MEM_LOAD_NODE)
        if tensor_type == "INPUT":
            node.input_tensor_loc = self.get_mem_type(mem_type)
            node.input_tensor_size = mem_size
        elif tensor_type == "OUTPUT":
            raise ValueError("Output tensors don't need to be loaded")
            #node.output_tensor_loc = self.get_mem_type(mem_type)
            #node.output_tensor_size = mem_size
        else:
            node.tensor_loc = self.get_mem_type(mem_type)
            node.tensor_size = mem_size
        return node

    def get_memory_store_node(self, layer_name: str, tensor_type: str, mem_type: str, mem_size: int) -> Node:
        node = self.get_node("MEM_STORE_NODE_" + layer_name + "_" + tensor_type, MEM_LOAD_NODE)
        if tensor_type == "INPUT":
            raise ValueError("Input tensors don't need to be stored")
            #node.input_tensor_loc = self.get_mem_type(mem_type)
            #node.input_tensor_size = mem_size
        elif tensor_type == "OUTPUT":
            node.output_tensor_loc = self.get_mem_type(mem_type)
            node.output_tensor_size = mem_size
        else:
            raise ValueError("Weight tensors don't need to be stored")
            #node.tensor_loc = self.get_mem_type(mem_type)
            #node.tensor_size = mem_size
        return node

    def get_comm_coll_node(self, layer_name: str, comm_type: str, comm_size: int) -> Node:
        node = self.get_node(
                f"COMM_COLL_NODE_{layer_name}_{comm_type}",
                COMM_COLL_NODE)
        node.comm_type = self.get_comm_type(comm_type)
        node.comm_size = comm_size
        return node

    def get_comm_node(self, is_send: bool, layer_name: str, comm_type: str, comm_size: int,
                           comm_src: int, comm_dst: int) -> Node:
        if is_send:
            node = self.get_node(
                    f"COMM_SEND_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_SEND_NODE)
        else:
            node = self.get_node(
                    f"COMM_RECV_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_RECV_NODE)
        node.comm_type = self.get_comm_type(comm_type)
        node.comm_src = comm_src
        node.comm_dst = comm_dst
        node.comm_size = comm_size
        comm_key = f"{node.comm_src}_{node.comm_dst}"
        if comm_key in self.comm_tag_dict:
            node.comm_tag = self.comm_tag_dict[comm_key]
        else:
            node.comm_tag = self.get_next_comm_tag()
            self.comm_tag_dict[comm_key] = node.comm_tag

        # check if SEND/RECV pair have same tags
        print(f"name: {node.name}, key: {comm_key}, tag: {node.comm_tag}")
        return node

    @staticmethod
    def get_comm_type(comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return INVALID_COMM
    
    @staticmethod
    def get_mem_type(mem_type: str) -> int:
        if mem_type == "LOCAL_MEMORY":
            return LOCAL_MEMORY
        elif mem_type == "REMOTE_MEMORY":
            return REMOTE_MEMORY
        elif mem_type == "STORAGE_MEMORY":
            return STORAGE_MEMORY
        return INVALID_MEMORY

    @staticmethod
    def add_parent(child_node: Node, parent_node: Node) -> None:
        child_node.parent.append(parent_node.id)

    def convert_tensor_parallel(self, f: TextIOWrapper, num_layers: int):
        layers: list[Layer] = self.get_layers(f)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                for idx, layer in enumerate(layers):
                    # Load input (for the first layer)
                    if idx == 0:
                        input_load_node = self.get_memory_load_node(
                            layer.name,
                            "INPUT",
                            layer.input_memory_loc,
                            layer.input_memory_size // self.num_npus
                        )
                        layer.input_memory_node = input_load_node
                        encode_message(g, input_load_node)

                    # Load weight
                    weight_load_node = self.get_memory_load_node(
                        layer.name,
                        "WEIGHT",
                        layer.weight_memory_loc,
                        layer.weight_memory_size // self.num_npus
                    )
                    layer.weight_memory_node = weight_load_node
                    encode_message(g, weight_load_node)

                    # Compute
                    comp_node = self.get_comp_node(layer.name, layer.comp_time // self.num_npus)
                    layer.comp_node = comp_node
                    if idx == 0:
                        self.add_parent(comp_node, input_load_node)
                    else:
                        self.add_parent(comp_node, layers[idx - 1].comm_node)
                    self.add_parent(comp_node, weight_load_node)
                    encode_message(g, comp_node)

                    # Communication (if required)
                    comm_coll_node = self.get_comm_coll_node(layer.name, layer.comm_type, layer.comm_size)
                    for j in range(self.num_dims):
                        comm_coll_node.involved_dim.append(True)
                    layer.comm_node = comm_coll_node
                    self.add_parent(comm_coll_node, comp_node)
                    encode_message(g, comm_coll_node)

                    # Store output
                    output_store_node = self.get_memory_store_node(
                        layer.name,
                        "OUTPUT",
                        layer.output_memory_loc,
                        layer.output_memory_size // self.num_npus
                    )
                    layer.output_memory_node = output_store_node
                    self.add_parent(output_store_node, comm_coll_node)
                    encode_message(g, output_store_node)

    def convert_pipeline_parallel(self, f: TextIOWrapper, num_layers: int):
        layers: list[Layer] = self.get_layers(f)

        if num_layers < self.num_npus: print("Warning! num_layers < self.num_npus, Some npus won't do anything!")
        layers_per_npu = num_layers // self.num_npus
        remain_layers = num_layers % self.num_npus
        current_layer_num = 0
        current_layer = None

        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                # Load input (for the first layer)
                if current_layer_num == 0:
                    input_load_node = self.get_memory_load_node(
                        layers[current_layer_num].name,
                        "INPUT",
                        layers[current_layer_num].input_memory_loc,
                        layers[current_layer_num].input_memory_size
                    )
                    layers[current_layer_num].input_memory_node = input_load_node
                    encode_message(g, input_load_node)
                # Receive input (from the previous layer in another npu)
                else:
                    receive_input_node = self.get_comm_node(
                        is_send=False,
                        layer_name=layers[current_layer_num].name,
                        comm_type=layers[current_layer_num].comm_type,
                        comm_size=layers[current_layer_num].input_memory_size,
                        comm_src=npu_id-1,
                        comm_dst=npu_id
                    )
                    layers[current_layer_num].comm_node = receive_input_node
                    encode_message(g, receive_input_node)

                for i in range(layers_per_npu + (1 if remain_layers > 0 else 0)):
                    # Load weight
                    weight_load_node = self.get_memory_load_node(
                        layers[current_layer_num].name,
                        "WEIGHT",
                        layers[current_layer_num].weight_memory_loc,
                        layers[current_layer_num].weight_memory_size
                    )
                    layers[current_layer_num].weight_memory_node = weight_load_node
                    encode_message(g, weight_load_node)

                    # Compute
                    comp_node = self.get_comp_node(layers[current_layer_num].name, layers[current_layer_num].comp_time)
                    layers[current_layer_num].comp_node = comp_node
                    if current_layer_num == 0:
                        self.add_parent(comp_node, input_load_node)
                    else:
                        if i == 0:
                            self.add_parent(comp_node, receive_input_node)
                        else:
                            self.add_parent(comp_node, layers[current_layer_num - 1].comp_node)
                    self.add_parent(comp_node, weight_load_node)
                    encode_message(g, comp_node)
                    
                    current_layer_num += 1

                if remain_layers > 0:
                    remain_layers -= 1
                
                current_layer_num -= 1
                if current_layer_num == (len(layers) - 1):
                    # Store output (for the last layer)
                    output_store_node = self.get_memory_store_node(
                        layers[current_layer_num].name,
                        "OUTPUT",
                        layers[current_layer_num].output_memory_loc,
                        layers[current_layer_num].output_memory_size
                    )
                    layers[current_layer_num].output_memory_node = output_store_node
                    self.add_parent(output_store_node, comp_node)
                    encode_message(g, output_store_node)
                    return
                else:
                    # Send output (to the next layer in another npu)
                    send_output_node = self.get_comm_node(
                        is_send=True,
                        layer_name=layers[current_layer_num].name,
                        comm_type=layers[current_layer_num].comm_type,
                        comm_size=layers[current_layer_num].output_memory_size,
                        comm_src=npu_id,
                        comm_dst=npu_id+1
                    )
                    layers[current_layer_num].comm_node = send_output_node
                    self.add_parent(send_output_node, comp_node)
                    encode_message(g, send_output_node)
                current_layer_num += 1

    def convert_hybrid_tensor_pipeline(self, f: TextIOWrapper, num_layers: int, num_npu_group: int):
        layers: list[Layer] = self.get_layers(f)

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        layers_per_group = num_layers // num_npu_group
        remain_layers = num_layers % num_npu_group

        layer_start = 0
        layer_end = 0
        for npu_group in range(num_npu_group):
            layer_start = layer_end
            layer_end = layer_start + layers_per_group + (1 if remain_layers > 0 else 0)
            for npu_offset in range(npus_per_group):
                npu_id = npu_group * npus_per_group + npu_offset
                output_filename = "%s.%d.eg" % (self.output_filename, npu_id)
                with open(output_filename, "wb") as g:
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size // npus_per_group
                        )
                        layers[layer_start].input_memory_node_list.append(input_load_node)
                        encode_message(g, input_load_node)
                    else:
                        # Receive input (from the previous layer in another npu group)
                        receive_input_node = self.get_comm_node(
                            is_send=False,
                            layer_name=layers[layer_start].name,
                            comm_type=layers[layer_start].comm_type,
                            comm_size=layers[layer_start].input_memory_size // npus_per_group,
                            comm_src=npu_id - npus_per_group,
                            comm_dst=npu_id
                        )
                        layers[layer_start].comm_node_list.append(receive_input_node)
                        encode_message(g, receive_input_node)
                
                    for layer_num in range(layer_start, layer_end):
                        # Load weight
                        weight_load_node = self.get_memory_load_node(
                            layers[layer_num].name,
                            "WEIGHT",
                            layers[layer_num].weight_memory_loc,
                            layers[layer_num].weight_memory_size // npus_per_group
                        )
                        layers[layer_num].weight_memory_node_list.append(weight_load_node)
                        encode_message(g, weight_load_node)

                        # Compute
                        comp_node = self.get_comp_node(layers[layer_num].name, layers[layer_num].comp_time // npus_per_group)
                        layers[layer_num].comp_node_list.append(comp_node)
                        if layer_num == layer_start:
                            if npu_group == 0:
                                self.add_parent(comp_node, input_load_node)
                            else:
                                self.add_parent(comp_node, receive_input_node)
                        else:
                            self.add_parent(comp_node, layers[layer_num - 1].comp_node_list[npu_offset])
                        self.add_parent(comp_node, weight_load_node)
                        encode_message(g, comp_node)
                    
                    if npu_group == (num_npu_group - 1):
                        # Store output (for the last layer)
                        output_store_node = self.get_memory_store_node(
                            layers[layer_end - 1].name,
                            "OUTPUT",
                            layers[layer_end - 1].output_memory_loc,
                            layers[layer_end - 1].output_memory_size // npus_per_group
                        )
                        layers[layer_end - 1].output_memory_node_list.append(output_store_node)
                        self.add_parent(output_store_node, comp_node)
                        encode_message(g, output_store_node)
                    else:
                        # Send output (to the next layer in another npu group)
                        send_output_node = self.get_comm_node(
                            is_send=True,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size // npus_per_group,
                            comm_src=npu_id,
                            comm_dst=npu_id + npus_per_group
                        )
                        layers[layer_end - 1].comm_node_list.append(send_output_node)
                        self.add_parent(send_output_node, comp_node)
                        encode_message(g, send_output_node)
            remain_layers -= 1

    def convert(self):
        with open(self.input_filename, "r") as f:
            first_line = f.readline().strip().split()
            parallelism_type = first_line[0]

            if len(first_line) == 3:
                assert(first_line[1] == "model_parallel_NPU_group:")
                num_npu_group = int(first_line[2])
            else:
                num_npu_group = 0

            second_line = f.readline().strip()
            num_layers = int(second_line)

            third_line = f.readline() # This is for the table header, so just ignore it

            if parallelism_type == "TENSOR":
                self.convert_tensor_parallel(f, num_layers)
            elif parallelism_type == "PIPELINE":
                self.convert_pipeline_parallel(f, num_layers)
            elif parallelism_type == "HYBRID_TENSOR_PIPELINE":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_hybrid_tensor_pipeline(f, num_layers, num_npu_group)
            else:
                raise ValueError(f"Unsupported parallelism type, {parallelism_type}")