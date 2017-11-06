# coding: utf-8
__author__ = 'Marcin Kowiel'

import numpy as np
import math
import networkx as nx


class Blob():
    def __init__(self, volume, density, min_box_o, max_box_o, max_point_box_o_list, local_maxi_o, skeleton, surface, children=None,
                 map=None):
        self.volume = volume
        self.density = density
        self.min_box_o = min_box_o
        self.max_box_o = max_box_o

        if not isinstance(max_point_box_o_list, list):
            self.max_point_box_o_list = [max_point_box_o_list]
        else:
            self.max_point_box_o_list = list(max_point_box_o_list)
        self.local_maxi_o = local_maxi_o
        self.skeleton = skeleton
        # dirty hack to fix empty skeleton however it may be a bug in skeletonisation
        if len(self.skeleton) == 0:
            self.skeleton.extend(self.local_maxi_o)
        self.surface = surface

        self.children = list(children) if children is not None else children
        self.map = map

        #self.validate()

    def __str__(self):
        return "min_box_o: {}, max_box_o: {}, max_point_box_o_list: {}, local_maxi_o: {}".format(
            self.min_box_o,
            self.max_box_o,
            self.max_point_box_o_list,
            self.local_maxi_o
        )

    @property
    def parts(self):
        return len(self.max_point_box_o_list)

    def validate(self):
        def validate_min_max(point_list, msg, eps=0.001):
            valid = True
            for point in point_list:
                if not (self.lte_tuple(self.sub_scalar(self.min_box_o, eps), point) and
                        self.lte_tuple(point, self.add_scalar(self.max_box_o, eps))):
                    print msg, "invalid, point:", point, 'min', self.min_box_o, 'max', self.max_box_o
                    valid = False
            return valid

        local_maxi_o_valid = validate_min_max(self.local_maxi_o, "local_maxi_o")
        skeleton_valid = validate_min_max(self.skeleton, "skeleton")
        surface_valid = validate_min_max(self.surface, "surface")
        return local_maxi_o_valid and skeleton_valid and surface_valid

    def get_symmetries(self, points):
        if isinstance(points, list) and len(points) > 0 and isinstance(points[0], (list, tuple)) and len(points[0]) == 3:
            points_list = points
        else:
            points_list = [points]
        return points_list

        if self.map is not None:
            return self.map.transform_all_rel_to_000(np.array(points_list)).tolist()
        else:
            #sym_points = []
            #symmetries =  [((1,1,1), (0,0,0))] #, ((-1,-1,-1), (0,0,0)), ((-1,1,1), (100,0,0)), ((1,-1,1), (0,100, 0))]
            #for mul_sym, add_sym in symmetries:
            #    for xyz in points_list:
            #        sym_points.append(self.mul_add_tuple(xyz, mul_sym, add_sym))
            return points_list

    @classmethod
    def mul_add_tuple(cls, a, mul, add):
        return (a[0] * mul[0] + add[0], a[1] * mul[1] + add[1], a[2] * mul[2] + add[2])

    @classmethod
    def lte_tuple(cls, a, b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]

    @classmethod
    def gte_tuple(cls, a, b):
        return a[0] >= b[0] and a[1] >= b[1] and a[2] >= b[2]

    @classmethod
    def add_tuple(cls, a, b):
        return (a[0] + b[0], a[1]+ b[1], a[2] + b[2])

    @classmethod
    def sub_tuple(cls, a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @classmethod
    def sub_scalar(cls, a, scalar):
        return (a[0] - scalar, a[1] - scalar, a[2] - scalar)

    @classmethod
    def add_scalar(cls, a, scalar):
        return (a[0] + scalar, a[1] + scalar, a[2] + scalar)

    @classmethod
    def min_tuple(cls, a, b):
        return (min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]))

    @classmethod
    def max_tuple(cls, a, b):
        return (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]))

    @classmethod
    def distance_sq_tuple(cls, a, b):
        x = a[0] - b[0]
        y = a[1] - b[1]
        z = a[2] - b[2]
        return x*x+y*y+z*z

    def is_box_close(self, other, distance):
        other_min_box = self.get_symmetries(other.min_box_o)
        other_max_box = self.get_symmetries(other.max_box_o)
        for other_min, other_max in zip(other_min_box, other_max_box):

            sorted_other_min = self.min_tuple(other_min, other_max)
            sorted_other_max = self.max_tuple(other_min, other_max)
            if self.lte_tuple(self.sub_scalar(self.min_box_o, distance), sorted_other_max) and self.gte_tuple(self.add_scalar(self.max_box_o, distance), sorted_other_min):
                #print 'BOX CLOSE:', self.min_box_o, self.max_box_o, 'close to', sorted_other_min, sorted_other_max
                return True

        return False

    def _is_close(self, self_list, other_list, other, distance):
        distance_sq = distance * distance

        other_min = self.sub_scalar(other.min_box_o, distance)
        other_max = self.add_scalar(other.max_box_o, distance)

        other_min = self.get_symmetries(other_min)
        other_max = self.get_symmetries(other_max)

        for point in self_list:
            i_sym = 0
            for other_min_box, other_max_box in zip(other_min, other_max):
                sorted_other_min = self.min_tuple(other_min_box, other_max_box)
                sorted_other_max = self.max_tuple(other_min_box, other_max_box)

                if self.gte_tuple(point, sorted_other_min) and self.lte_tuple(point, sorted_other_max):
                    for point_other in other_list:
                        point_other_sym = self.get_symmetries(point_other)[i_sym]
                        if self.distance_sq_tuple(point, point_other_sym) <= distance_sq:
                            #print 'MAXI CLOSE', math.sqrt(self.distance_sq_tuple(point, point_other_sym)), distance, \
                            #    point, point_other_sym
                            return True
                i_sym += 1
        return False

    def is_local_maxi_close(self, other, distance):
        return self._is_close(self.local_maxi_o, other.local_maxi_o, other, distance)

    def is_skeleton_close(self, other, distance):
        return self._is_close(self.skeleton, other.skeleton, other, distance)

    def is_mergeable_with(self, other, distance=1.55):
        return self.is_box_close(other, distance) and (self.is_local_maxi_close(other, distance) or self.is_skeleton_close(other, distance))

    @classmethod
    def _merge_list(cls, others):
        """
        Constructor

        :param others: list of blobs
        :return: blob with children
        """
        min_box_o = others[0].min_box_o
        max_box_o = others[0].max_box_o
        max_point_box_o_list = []
        local_maxi_o = []
        skeleton = []
        surface = []
        children = []
        volume = 0.0
        density = 0.0
        for blob in others:
            if (volume + blob.volume) == 0.0:
                density = 0.0
            else:
                density = (float(volume) * density + float(blob.volume) * blob.density) / float(volume + blob.volume)
            volume += blob.volume
            min_box_o = cls.min_tuple(min_box_o, blob.min_box_o)
            max_box_o = cls.max_tuple(max_box_o, blob.max_box_o)
            max_point_box_o_list.extend(blob.max_point_box_o_list)
            local_maxi_o.extend(blob.local_maxi_o)
            skeleton.extend(blob.skeleton)
            surface.extend(blob.surface)
            if blob.children is not None:
                children.extend(blob.children)
            else:
                children.append(blob)

        return Blob(volume, density, min_box_o, max_box_o, max_point_box_o_list, local_maxi_o, skeleton, surface, children=children)

    @classmethod
    def merge(cls, blobs, distance):
        # blob_labels will have unique label of blob
        blob_labels = list(range(1, len(blobs)+1))
        for i_blob, first_blob in enumerate(blobs):
            for j_blob, second_blob in enumerate(blobs[i_blob+1:]):
                if first_blob.is_mergeable_with(second_blob, distance):
                    for pos in range(len(blob_labels)):
                        if blob_labels[pos] == blob_labels[i_blob+j_blob+1]:
                            blob_labels[pos] = blob_labels[i_blob]

        blobs_to_merge = {}

        for i_blob, blob in enumerate(blobs):
            blobs_to_merge.setdefault(blob_labels[i_blob], []).append(blob)

        merged_blobs = []

        for key, blob_list in blobs_to_merge.iteritems():
            print 'MERGE', key, 'using', len(blob_list), 'blobs'
            for blob in blob_list:
                print '    ', blob.volume, blob.density, 'loc_m', len(blob.local_maxi_o), 'skel', len(blob.skeleton)
            merged_blobs.append(cls._merge_list(blob_list))

        merged_blobs = sorted(merged_blobs, key=lambda x: x.volume, reverse=True)

        return merged_blobs

    def graph(self, edge_distance, point_list):
        distance_sq = edge_distance * edge_distance
        graph = nx.Graph()
        for i_point, point_i in enumerate(point_list):
            graph.add_node(i_point, xyz=point_i)

            for j_point, point_j in enumerate(point_list[i_point + 1:], i_point + 1):
                point_distance_sq = self.distance_sq_tuple(point_i, point_j)

                if point_distance_sq < distance_sq:
                    length = math.sqrt(point_distance_sq)
                    graph.add_edge(i_point, j_point, length=length)

        subgraphs = [sg for sg in nx.connected_component_subgraphs(graph)]
        subgraph_num = len(subgraphs)

        while subgraph_num > 1:
            subgraph_ids = np.array(range(subgraph_num))

            for i in range(subgraph_num):
                # check if subgraph not connected to any other subgraph
                if len(subgraph_ids[subgraph_ids == i]) == 1:
                    min_distance_sq = float("inf")
                    min_n_j = -1
                    min_j = -1
                    min_n_i = -1

                    for n_i in subgraphs[i]:
                        for j in range(subgraph_num):
                            if i != j:
                                for n_j in subgraphs[j]:
                                    point_distance_sq = self.distance_sq_tuple(point_list[n_i],
                                                                               point_list[n_j])

                                    if point_distance_sq < min_distance_sq:
                                        min_distance_sq = point_distance_sq
                                        min_n_j = n_j
                                        min_j = j
                                        min_n_i = n_i

                    if (min_n_j != -1 and  min_j != -1 and min_n_i != -1):
                        length = math.sqrt(min_distance_sq)
                        graph.add_edge(min_n_i, min_n_j, length=length)
                        subgraph_ids[min_j] = i
                    else:
                        raise Exception("should not happen")

            subgraphs = [sg for sg in nx.connected_component_subgraphs(graph)]
            subgraph_num = len(subgraphs)

        return graph

    def local_max_graph(self, edge_distance):
        return self.graph(edge_distance, self.local_maxi_o)

    def skeleton_graph(self, edge_distance):
        return self.graph(edge_distance, self.skeleton)

    def prune_deg_3(cls, graph, min_length=(2.5*0.2)):
        cycles = nx.cycle_basis(graph)
        for cycle in cycles:
            if len(cycle) == 3:
                node0 = cycle[0]
                node1 = cycle[1]
                node2 = cycle[2]
                print node0, node1, node2, len(graph.nodes())
                l1 = graph[node0][node1]['length'] if graph.has_edge(node0, node1) else -1
                l2 = graph[node1][node2]['length'] if graph.has_edge(node1, node2) else -1
                l3 = graph[node2][node0]['length'] if graph.has_edge(node2, node0) else -1
                if 0 < l1 < min_length and 0 < l2 < min_length and 0 < l3 < min_length:
                    if l1 <= l2 and l1 <= l3:
                        graph.remove_edge(node0, node1)
                    elif l2 <= l1 and l2 <=l3:
                        graph.remove_edge(node1, node2)
                    elif l3 <= l1 and l3 <=l2:
                        graph.remove_edge(node2, node0)
        return graph

    @classmethod
    def graph_descriptors(cls, graph):
        subgraph = []
        eccentricity = []
        for sub in nx.connected_component_subgraphs(graph):
            subgraph.append(sub)
            eccentricity.append(nx.eccentricity(sub))

        diameter = 0
        radius = 0
        center = 0
        periphery = 0
        if len(subgraph) > 0:
            diameter = sum((nx.diameter(sub, ecc) for sub, ecc in zip(subgraph, eccentricity))) + len(subgraph) - 1
            radius = sum((nx.radius(sub, ecc) for sub, ecc in zip(subgraph, eccentricity))) + len(subgraph) - 1

            center = sum((len(nx.center(sub, ecc)) for sub, ecc in zip(subgraph, eccentricity)))
            periphery = sum((len(nx.periphery(sub, ecc)) for sub, ecc in zip(subgraph, eccentricity)))

        nodes_len = graph.number_of_nodes()
        edges_len = graph.number_of_edges()
        cycles = nx.cycle_basis(graph)
        deg_hist = nx.degree_histogram(graph)
        closeness = nx.closeness_centrality(graph, distance='length')

        dsc = {
            'nodes': nodes_len,
            'edges': edges_len,
            'avg_degree': 2.0*edges_len/nodes_len,
            'density': nx.density(graph),
            'deg_0': deg_hist[0] if len(deg_hist) > 0 else 0,
            'deg_1': deg_hist[1] if len(deg_hist) > 1 else 0,
            'deg_2': deg_hist[2] if len(deg_hist) > 2 else 0,
            'deg_3': deg_hist[3] if len(deg_hist) > 3 else 0,
            'deg_4': deg_hist[4] if len(deg_hist) > 4 else 0,
            'deg_5_plus': sum(deg_hist[5:]) if len(deg_hist) > 5 else 0,
            'diameter': diameter,
            'radius': radius,
            'center':  center,
            'periphery': periphery,
            'average_clustering': nx.average_clustering(graph),
            'graph_clique_number': nx.graph_clique_number(graph),
            #'k_components': nx.algorithms.k_components(graph),

            'cycles': len(cycles),
            'cycle_3': len([x for x in cycles if len(x) == 3]),
            'cycle_4': len([x for x in cycles if len(x) == 4]),
            'cycle_5': len([x for x in cycles if len(x) == 5]),
            'cycle_6': len([x for x in cycles if len(x) == 6]),
            'cycle_7': len([x for x in cycles if len(x) == 7]),
            'cycle_8_plus': len([x for x in cycles if len(x) > 7]),

            'closeness_000_002': len([x for x in closeness.values() if x <= 0.02]),
            'closeness_002_004': len([x for x in closeness.values() if 0.02 < x <= 0.04]),
            'closeness_004_006': len([x for x in closeness.values() if 0.04 < x <= 0.06]),
            'closeness_006_008': len([x for x in closeness.values() if 0.06 < x <= 0.08]),
            'closeness_008_010': len([x for x in closeness.values() if 0.08 < x <= 0.10]),
            'closeness_010_012': len([x for x in closeness.values() if 0.10 < x <= 0.12]),
            'closeness_012_014': len([x for x in closeness.values() if 0.12 < x <= 0.14]),
            'closeness_014_016': len([x for x in closeness.values() if 0.14 < x <= 0.16]),
            'closeness_016_018': len([x for x in closeness.values() if 0.16 < x <= 0.18]),
            'closeness_018_020': len([x for x in closeness.values() if 0.18 < x <= 0.20]),
            'closeness_020_030': len([x for x in closeness.values() if 0.20 < x <= 0.30]),
            'closeness_030_040': len([x for x in closeness.values() if 0.30 < x <= 0.40]),
            'closeness_040_050': len([x for x in closeness.values() if 0.40 < x <= 0.50]),
            'closeness_050_plus': len([x for x in closeness.values() if 0.50 < x]),
        }
        return dsc
