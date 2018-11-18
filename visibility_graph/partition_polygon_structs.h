struct edge_occluders{
	int _edge_pts[] = {};

	int get_num_edges(){
		return len(_edge_pts)/2;
	}
	int get_pt1(int edge){
		return this._edge_pts[edge*2];
	}
	int get_pt2(int edge){
		return this._edge_pts[edge*2+1];
	}
	void add_edge(int i, j){
		printf("added edge: "+ itoa(i) + ", " + itoa(j) +"\n");
		//allways store smallest pt number first
		if (i>j){
			append(_edge_pts, j);
			append(_edge_pts, i);
		}
		else if (j>i){
			append(_edge_pts, i);
			append(_edge_pts, j);
		}	
		/*else{
			printf("invalid edge- pt %i to %i", i, j);
		}*/
	}

	void remove_edge(int index){
		pop(this._edge_pts, index*2);
		pop(this._edge_pts, index*2);
	}	

	void get_edge_pts(int n, pt0, pt1){
		pt0 = this._edge_pts[n*2];
		pt1 = this._edge_pts[n*2+1];
	}

	int get_edge_from_pts(int pt0, pt1){
		int e0 = min(pt0, pt1);
		int e1 = max(pt0, pt1);
		int n_edges = this->get_num_edges();
		int edge0, edge1;
		int i=0;
		while(i< n_edges){
			this->get_edge_pts(i, e0, e1);
			if(e0==pt0 && e1==pt1){
				return i;
			}
			i++;
		}
		//printf("can't find edge: "+ itoa(pt0) + ", " + itoa(pt1) +"\n");
		return -1;
	}
	
	void remove_edge_from_pts(int pt0, pt1){
		int edge = this->get_edge_from_pts(pt0, pt1);
		this->remove_edge(edge);
	}

	int check_point_in_edge(int pt, edge){
		if (this._edge_pts[edge*2] == pt || this._edge_pts[edge*2+1] == pt )return 1;
		return 0;
	}

	void clear(){
		_edge_pts[] = {};
	}

	int get_edge_from_point(int pt, offset){
		int index = find(_edge_pts, pt, offset);
		index -= index%2;
		return index;
	}
	//returns the other point for an edge
    int get_adjacent(int index, pt_num){
        int e_pt0, e_pt1;
        this->get_edge_pts(index, e_pt0, e_pt1);

		if (pt_num == e_pt0) return e_pt1;
    	else if(pt_num == e_pt1) return e_pt0;
    	printf("error on trying to get_adjacent for edge: %g, %g, pt: %g", e_pt0, e_pt1, pt_num);
    	return -1;
    	
	}


	//START OF COPY from OpenEdge class in pyvisgraph,which sorts the edges by size 
	//adds another edge (e1,e2) to the list. p1, p2
    void insert(int p1, p2, e1, e2){
    	// rewrite for _open_edges.insert(self._index(p1, p2, edge), edge);
    	int index = this->_index(p1, p2, e1, e2);
        insert(this._edge_pts, index*2, p2); 
        insert(this._edge_pts, index*2, p1); 
	}

    void  delete(int p1, p2, ){
        int index = this->_index(p1, p2, edge) - 1;
        if (this->compareEdges(index == edge) ){
            this->remove_edge(index);
        }
	}
    void  smallest(int pt1, pt2) { 
        get_edge_pts(0, pt1, pt2);
	}
//TODO IMPLEMENT
    void _less_than(int p1, p2, edge1_1, edge1_2, edge2_1, edge2_2){
        //"""Return True if edge1 is smaller than edge2, False otherwise."""
        if (edge1 == edge2) return 0;
        if (!edge_intersect(p1, p2, edge2) ) return 1;
        edge1_dist = point_edge_distance(p1, p2, edge1);
        edge2_dist = point_edge_distance(p1, p2, edge2);
        if (edge1_dist > edge2_dist) return 0;
        if (edge1_dist < edge2_dist) return 1;
        //If the distance is equal, we need to compare on the edge angles.
        if (edge1_dist == edge2_dist){
        	if (this->check_point_in_edge(this->get_pt_a(edge1), edge2) )same_point = edge1.p1;
        	else same_point = edge1.p2;
        }
        
        float angle_edge1 = angle2(p1, p2, this->get_adjacent(edge1, same_point));
        float angle_edge2 = angle2(p1, p2, this->get_adjacent(edge2, same_point));
        if (angle_edge1 < angle_edge2) return 1;

        return 0;
	}
//TODO IMPLEMENT
    void _index( int p1, p2, edge){
        lo = 0;
        hi = len(self._open_edges);
        while (lo < hi)
            mid = int((lo+hi)/2); //originally python '(lo+hi)//2'
            if (this->_less_than(p1, p2, edge, this._open_edges[mid] ) ) hi = mid;
            else lo = mid + 1;
        return lo;
		}
//END OF COPY
	void print_pts(){
		printf("%g\n", _edge_pts);
	}
}


struct visibility_graph{
	int _visibility_graph[]={}; //two dimensional array of visibility
	int _primnum =-1; //current polygon
	int _size=1;
	int _INF = 1000000.0; //comput this at init
	int _COLIN_TOLERANCE = 10;
	int _DEBUG = 1; 
	int CCW = 1; // from pyvisgraph, not consistently used here 
	int CW = -1;
	int COLLINEAR = 0;	
	//int COLIN_TOLERANCE = 10;
	int T = 10000000000; 
	float T2 = 10000000000.0;
	edge_occluders _sweep_edges = {}; 


	int _E[];
	vector2 _pts_loc_pos[]; 
	int _polygon_pts[]; 
	vector _centroid = 0; 
	vector4 _rot_to_local = 1;
	
	int _current_sweep_pt;
	vector2 _current_sweep_local_pos;

	//float T = pow(10.0, COLIN_TOLERANCE);
	//float T2 = pow(10.0, COLIN_TOLERANCE);
	//TODO implement sweep edges functions:
	//is_intersecting_vector(edge, vector),update_sweep_edges_for_point(sweep_pt, edge_pt) is_ccw(edge)


	void set_xform_from_N(vector centroid; vector poly_N){
		_centroid = centroid;
		_rot_to_local = dihedral(poly_N, set(0,1,0));
	}

	//returns a 2d vector in y plane (previously 3d vector)
	vector2 xform_to_local_space(vector p){ 
		p-=_centroid;
		p = qrotate(_rot_to_local, p);
		return set(p.x, p.z);
	}

	vector2 get_pt_local_pos(int pt){
		vector pos = point(0, "P", pt);
		vector2 local_pos = this->xform_to_local_space(pos);
		return local_pos;
	}

	void init(int primnum; vector centroid; vector poly_N; int debug){
		this._primnum = primnum;
		this._polygon_pts = primpoints(0, primnum);
		this._size= len(_polygon_pts);
		this->set_xform_from_N(centroid, poly_N);
		this._pts_loc_pos = {};
		this._DEBUG = debug;
		foreach(int pt; _polygon_pts){
			vector2 pt_pos_local = this->get_pt_local_pos(pt);
			append(this._pts_loc_pos, pt_pos_local);
		}
	}

	void set_visibility(int i; int j; int value){
		_visibility_graph[this._size*i+j] = value;
	}

	int get_visibility(int i; int j){
		return _visibility_graph[this._size*i+j];
	}	

	void clear_sweep_edges(){
		_sweep_edges->clear();
	}

	//_sweep_edges functions
	void add_edge_to_sweep(int start_pt, end_pt){
		_sweep_edges->insert(start_pt, end_pt); //python version (replaced _sweep_edges->add_edge(start_pt, end_pt);)
	}

	void remove_sweep_edges_for_pt(int ref_pt){
		int edge_index = get_edge_from_point(edge_pt);
		while(edge_index>=0){
			_sweep_edges->remove_edge(edge_index);
			 edge_index = get_edge_from_point(edge_pt);
		}
	}


	void debug_polyline(int pt0, pt1; vector col){
		int new_prim = addprim(0, "polyline");
		addvertex(0, new_prim, pt0);
		addvertex(0, new_prim, pt1);
		setprimattrib(0, "Cd", new_prim, col);
	}

	void debug_polyline(vector pos0, pos1, col){
		int pt0 = addpoint(0, pos0);
		int pt1 = addpoint(0, pos1);
		this->debug_polyline(pt0, pt1, col);
	}	
	
	int query_segment_segment_intersect(vector p0, p1, p2, p3){

		//intersection test for two edges
		//taken from https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
		//

	    float s2_x = p3.x - p2.x;     
	    float s2_z = p3.z - p2.z;

	    float s, t;
	    s = (-s1_z * (p0.x - p2.x) + s1_x * (p0.z - p2.z)) / (-s2_x * s1_z + s1_x * s2_z);
	    t = ( s2_x * (p0.z - p2.z) - s2_z * (p0.x - p2.x)) / (-s2_x * s1_z + s1_x * s2_z);

	    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
	    {
	    	if(this._DEBUG){
	    		//t*=1;
	    		int intersection_pt = addpoint(0, set(p0.x + (t * s1_x), 0,  p0.z + (t * s1_z)));
	    		int sweep_line_with_offset =addpoint(0, set(p0.x + (.05 * s1_x), 0,  p0.z + (.05 * s1_z)) );
				this->debug_polyline(intersection_pt, sweep_line_with_offset, set(1,0,0) );
				this->debug_polyline(pt2, pt3, set(0,1,0) );
	    	}
	    	return 0;
		}
	}

	//get next and previous point on the polygon into given ints
	void get_neighbours(int index,  pt_prev, pt_next){
		pt_prev = _polygon_pts[(index-1)%this._size];
		pt_next = _polygon_pts[(index+1)%this._size];
	}

	//get next and previous point on the polygon into a given array
	void get_neighbours(int index,  nbrs[]){
		nbrs ={};
		append(nbrs, _polygon_pts[(index-1)%this._size] );
		append(nbrs, _polygon_pts[(index+1)%this._size] );
	}

	//finds all points connected to pt that is part of the polygon
	void get_connected(int pt, connected_pts_on_prim[]){
		int nbrs[] = neighbours(0, pt);
		connected_pts_on_prim = {};
		foreach(int p; nbrs){
			if (find(this._polygon_pts, p) ) append(connected_pts_on_prim, p);
		}

	}


	int ccw(vector2 A, B, C){
		//1 = ccw, -1 =cw, 0 = colinear 
		int a = int( (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)*T)/T2;  //to deal with truncation errors original version ends with *T)/T2;;
		if(a>0) return 1;
		if(a<0) return -1;
		return 0; 
	}

	int ccw(int pt_a, pt_b, pt_c){
		vector2 A = this->get_pt_local_pos(pt_a);
		vector2 B = this->get_pt_local_pos(pt_b);
		vector2 C = this->get_pt_local_pos(pt_c);
		int ccw_result =  this->ccw(A, B, C);
		return ccw_result;
	}
	int on_segment(vector2 p, q, r){
		if (q.x<=max(p.x, r.x) && q.x>= min(p.x, r.x)  ) {
			if (q.y<=max(p.y, r.y) && q.y>= min(p.y, r.y)  ){
				return 1;
			}
		}
		return 0;
	}

	int edge_intersect(vector2 p1, q1, p2, q2){
		int o1 = this->ccw(p1, q1, p2);
		int o2 = this->ccw(p1, q1, q2);	
		int o3 = this->ccw(p2, q2, p1);	
		int o4 = this->ccw(p2, q2, q1);

		if(o1 != o2 && o3!= o4) {
			//printf ("gen case\n");
			//printf ("%g, %g, %g, %g\n", o1, o2, o3, o4); 
			return 1;
		}
		if(o1 == COLLINEAR  && this->on_segment(p1, p2, q1)){return 1;}
		if(o2 == COLLINEAR  && this->on_segment(p1, q2, q1)){return 1;}
		if(o3 == COLLINEAR  && this->on_segment(p2, p1, q2)){return 1;}
		if(o4 == COLLINEAR  && this->on_segment(p2, q1, q2)){return 1;}
		return 0;
	}

	int edge_in_polygon(int p0, p1){
		int neighbour_pts[] = neighbours(0, p0);
		if (find(neighbour_pts, p1)){return 1;}
		return 0;
	}

	//sorts all points in order after sweep search around point pt
	void sort_pts_around_point(const int pt_p; int sorted_pts[]){
	    //Sort points after angle around pt, starting in x dir, closeness where values are equal
	    sorted_pts = _polygon_pts;
	    float angles[];
	    vector pos_p = point(0, "P", pt_p);
	    //get angles
	    foreach(int pt_w; sorted_pts){
	        printf("pt_w: %g  pt_p: %g\n", pt_w, pt_p);
	        if (pt_w==pt_p) 
	            append(angles, -360);//force first
	        else{
	            vector pos_w = point(0, "P", pt_w);
	            vector pos_pw = pos_w - pos_p;
	            vector2 pos_pw_local =  this->xform_to_local_space(pos_pw);
	            float angle_w= atan2(pos_pw_local.x, pos_pw_local.y); //(needs check of sign?) flipped axixes gives angle from x axis as algorithm asks for
	            if (_DEBUG==1){printf("pt %g, pt_w %g, angle_w: %g\n", pt_p, pt_w, degrees(angle_w));}
	            append(angles, angle_w);

	        }
	    }
	    if(_DEBUG) printf("angles %g\n", angles);
	    //sort angles
	    int angle_sort_keys[]= argsort(angles);
	    sorted_pts = reorder(sorted_pts, angle_sort_keys);
	    if(_DEBUG) printf("sorted_pts: %g \n", sorted_pts); 
	}

	// test sweep line against all edges 
	void init_sweep(int sweep_pt, sorted_pts[]){
		//TODO move sorting from scene wrangler to struct
		//update sweep pt
		_current_sweep_pt = sweep_pt;
		vector2 _current_sweep_local_pos = this->get_pt_local_pos(sweep_pt);
		vector2 sweep_axis = set(_INF, 0.0);
		vector2 local_sweep_axis_pos = _current_sweep_local_pos+sweep_axis;
		vector2 local_sweep_axis_pos_preview = _current_sweep_local_pos+ set(2,0) ;

		if(_DEBUG){
			printf("adding debug axis line (cyan)\n");
			this->debug_polyline(_current_sweep_local_pos, local_sweep_axis_pos_preview, set(0,1,1) );
		}		

		//check all edges for intersection with sweep half line
		for (int i=0; i<len(_polygon_pts); i++){
			int edge_pt2 = _polygon_pts[i]; 
			int edge_pt1 = _polygon_pts[(i-1)%this._size]; 
			vector2 edge_pos1 = this->get_pt_local_pos(edge_pt1);
			vector2 edge_pos2 = this->get_pt_local_pos(edge_pt2);

			printf("testing pt, %i pt_prev, %i\n", edge_pt1, edge_pt2);
		 	// neighbour edges
			if ( edge_pt2==_current_sweep_pt || edge_pt1==_current_sweep_pt ){
				if(_DEBUG ){
					printf("\tadding neighbour edge (purple)\n");
					this->debug_polyline(edge_pt2, edge_pt1, set(1,0,1) );
				}
			}
			// half axis intersection edges
			else if(this->edge_intersect(_current_sweep_local_pos, local_sweep_axis_pos, edge_pos1, edge_pos2) ){
					printf("found intersection: %g, %g \n", edge_pt1, edge_pt2);
					//check if edge end points is on axis
					if(this->on_segment(_current_sweep_local_pos, edge_pos1, local_sweep_axis_pos )) continue;	
					if(this->on_segment(_current_sweep_local_pos, edge_pos2, local_sweep_axis_pos )) continue;
					this->add_edge_to_sweep(edge_pt1, edge_pt2);
					if(_DEBUG){
						this->debug_polyline( set(0,1,0) + _current_sweep_local_pos, set(0,1,0) + local_sweep_axis_pos_preview, set(1,1,0) );
						this->debug_polyline(set(0,1,0) + edge_pos1, set(0,1,0) + edge_pos2, set(1,1,0) );
						printf("\tadding init intersection edge (yellow) %g, %g\n",edge_pt1 , edge_pt2);
						this->debug_polyline(edge_pos1, edge_pos2, set(1,1,0) );
					}
			}
			// rest
			else{
			  	if(_DEBUG){
					printf("\tnon initialized edge (green) %g, %g\n",edge_pt1 , edge_pt2);
					this->debug_polyline(edge_pt1, edge_pt2, set(0,1,0) );
				}
			}	
		}	
	}
	//removes edges from new sweep check point
	void update_sweep_edges_remove_cw_from_p(int point, p){
        /*(open_edges assumed to be uquivalent to my _sweep_edges, needs checking though): 
        if open_edges:
        for edge in graph[p]:
            if ccw(point, p, edge.get_adjacent(p)) == CW:
                open_edges.delete(point, p, edge)
                */
		if (this._sweep_edges->get_num_edges()==0){return;}
		int nbrs[] = {};
		this->get_neighbours(p, nbrs);
        foreach(int nbr; nbrs){ //  //removed: int e= 0; e*2<this._sweep_edges->get_num_edges(); e++){  
        	int y = 9;
			if(CW == this->ccw(point, p,  nbr) ){  //the funciton call should (probably) get the neighbour point
				this._sweep_edges->remove_edge_from_pts(p, nbr);
			}
        }
	}
	//removes edges from new sweep check point
	void update_sweep_edges_add_cw_from_p(int point, p){
        /*(open_edges assumed to be uquivalent to my _sweep_edges, needs checking though): 
        for edge in graph[p]:
            if (point not in edge) and ccw(point, p, edge.get_adjacent(p)) == CCW:
                open_edges.insert(point, p, edge)
                */
		if (this._sweep_edges->get_num_edges()==0){return;}
		int nbrs[] = {};
		this->get_neighbours(p, nbrs);
        foreach(int nbr; nbrs){ //  //removed: int e= 0; e*2<this._sweep_edges->get_num_edges(); e++){  
			if(point != nbr && CCW == this->ccw(point, p,  nbr) ){  //the funciton call should (probably) get the neighbour point
				this->add_edge_to_sweep(p, nbr);
			}
        }
	}
	int check_pt_visible(int point, p, is_visible, prev, prev_visible){
			//4. Check if p is visible from point
	        // Check if p is visible from point
	        is_visible = 0;
	        // ...Non-collinear points
	        if (prev == 0 || this->ccw(point, prev, p) != COLLINEAR || this->on_segment(point, prev, p)==0 ){
	            if (this._sweep_edges->get_num_edges() == 0) is_visible = 1;
	            else if (! this->edge_intersect(point, p, this._sweep_edges->get_pt1(0), this._sweep_edges->get_pt2(0)  ) ) is_visible = 1;  //this._sweep_edges->get_pt_a(0) gives the first pt in the smallest edge 
	        // ...For collinear points, if previous point was not visible, p is not
        	}		
	        else if (!prev_visible)	 is_visible = 0;
	        // ...For collinear points, if previous point was visible, need to check
	        // that the edge from prev to p does not intersect any open edge.
	        else{
	            is_visible = 1;
	            for (int e=0; e*2< this._sweep_edges->get_num_edges(); e++){
	                if ( this._sweep_edges->check_point_in_edge(prev, e)  && this->edge_intersect(prev, p, this._sweep_edges->get_pt1(e), this._sweep_edges->get_pt2(e) )) {
	                    is_visible = 0;
	                    break;
	                }
	            }
	            if (is_visible &&  this->edge_in_polygon(prev, p)) is_visible = 0;
            }
            return is_visible;
	        // Check if the visible edge is interior to its polygon
	        //TODO, translate to new code
	        //if( is_visible &&  ! find(graph.get_adjacent_points(point), p ) ) is_visible = 1- edge_in_polygon(point, p, graph); //TODO implement graph.get_adjacent_points(point) 
	        

	        /////PYTHON REF CODE
	        /*
			//4. Check if p is visible from point
	        # Check if p is visible from point
	        is_visible = False
	        # ...Non-collinear points
	        if prev is None or ccw(point, prev, p) != COLLINEAR or not on_segment(point, prev, p):
	            if len(open_edges) == 0:
	                is_visible = True
	            elif not edge_intersect(point, p, open_edges.smallest()):
	                is_visible = True
	        # ...For collinear points, if previous point was not visible, p is not
	        elif not prev_visible:
	            is_visible = False
	        # ...For collinear points, if previous point was visible, need to check
	        # that the edge from prev to p does not intersect any open edge.
	        else:
	            is_visible = True
	            for edge in open_edges:
	                if prev not in edge and edge_intersect(prev, p, edge):
	                    is_visible = False
	                    break
	            if is_visible and edge_in_polygon(prev, p, graph):
	                    is_visible = False
	        # Check if the visible edge is interior to its polygon
	        if is_visible and p not in graph.get_adjacent_points(point):
	            is_visible = not edge_in_polygon(point, p, graph)
			//
			*/
        	////////////////////
	}
	//retrieve all visible points from the point pt and set it as an attrib
	void visibile_vertices(int point){
	    int visible[] = {};
	    int sorted_pts[];                      //add visible points to array W
	    // 1. sort points after angle
	    this->sort_pts_around_point( point, sorted_pts); 
	    // 2. setup visibility sweeping by loading edge_occluders-'_sweep_edges'
	    this->init_sweep( point, sorted_pts);     
	    int i = 0;
		int prev = -1;
        int prev_visible = 0;
        i==0;
	    foreach(int p; sorted_pts){
	    	i++;
	    	printf("iterate\n");
	    	_sweep_edges->print_pts();
	    	if(p==point)continue;
	        // 3. TODO Update _sweep_edges - remove clock wise edges incident on p 
    		this->update_sweep_edges_remove_cw_from_p( point, p);
	       _sweep_edges->print_pts();
	       //4. Check if p is visible from point
	        int is_visible = this->check_pt_visible(point, p, is_visible, prev, prev_visible);
	        if(is_visible) append(visible, p);
	        //5. # Update open_edges - Add counter clock wise edges incident on p
	        this->update_sweep_edges_add_cw_from_p(point, p); //TODO implement
	        _sweep_edges->print_pts();
	        prev = p;
	        prev_visible = is_visible;
	    }
	    setpointattrib(0, "vis_pts", point, visible );
	    printf("end visibility sweeping\n");

	}

/*
	void visible_vertices(point):
		// 1. sort points after angle
	    points.sort(key=lambda p: (angle(point, p), edge_distance(point, p)))
		// 2. setup visibility sweeping by loading edge_occluders-'_sweep_edges'
	    # Initialize open_edges with any intersecting edges on the half line from
	    # point along the positive x-axis
	    open_edges = OpenEdges()
	    point_inf = Point(INF, point.y)
	    for edge in edges:
	        if point in edge: continue
	        if edge_intersect(point, point_inf, edge):
	            if on_segment(point, edge.p1, point_inf): continue
	            if on_segment(point, edge.p2, point_inf): continue
	            open_edges.insert(point, point_inf, edge)

	    visible = []
	    prev = None
	    prev_visible = None
	    for p in points:
	        if p == point: continue
	        #if scan == 'half' and angle(point, p) > pi: break

	        //3. Update _sweep_edges - remove clock wise edges incident on p 
	        # Update open_edges - remove clock wise edges incident on p

			//4. Check if p is visible from point
	        # Check if p is visible from point
	        is_visible = False
	        # ...Non-collinear points
	        if prev is None or ccw(point, prev, p) != COLLINEAR or not on_segment(point, prev, p):
	            if len(open_edges) == 0:
	                is_visible = True
	            elif not edge_intersect(point, p, open_edges.smallest()):
	                is_visible = True
	        # ...For collinear points, if previous point was not visible, p is not
	        elif not prev_visible:
	            is_visible = False
	        # ...For collinear points, if previous point was visible, need to check
	        # that the edge from prev to p does not intersect any open edge.
	        else:
	            is_visible = True
	            for edge in open_edges:
	                if prev not in edge and edge_intersect(prev, p, edge):
	                    is_visible = False
	                    break
	            if is_visible and edge_in_polygon(prev, p, graph):
	                    is_visible = False
	        # Check if the visible edge is interior to its polygon
	        if is_visible and p not in graph.get_adjacent_points(point):
	            is_visible = not edge_in_polygon(point, p, graph)
			//

	        if is_visible: visible.append(p)

			//5. 
	        # Update open_edges - Add counter clock wise edges incident on p
	        for edge in graph[p]:
	            if (point not in edge) and ccw(point, p, edge.get_adjacent(p)) == CCW:
	                open_edges.insert(point, p, edge)

	        prev = p
	        prev_visible = is_visible
	    return visible

	    */


	void xform_all_pts_to_local_space(){
			foreach(int pt; this._polygon_pts){
				vector local_pos = this->get_pt_local_pos(pt);
				setpointattrib(0, "P", pt, local_pos);
			}
		return;

		}

}
