#include <Python.h>
#include <numpy/arrayobject.h>
#include <AlphaExpansion_2D_4C.h>
#include <AlphaExpansion_2D_4C_MT.h>
#include <AlphaExpansion_2D_8C.h>

typedef AlphaExpansion_2D_4C<int, float, float> Expansion_2d_4c;
typedef AlphaExpansion_2D_4C_MT<int, float, float> Expansion_2d_4c_mt;
typedef AlphaExpansion_2D_8C<int, float, float> Expansion_2d_8c;


/*
Extract the solution from a grid as a numpy int array.
Grid must alread have been solved for this to work.
Returns a single 1D numpy int array where result[x+y*width] is the binary label assigned to node x,y
*/
static PyObject* labels_to_array(int* labels, int width, int height)
{
	//create an array to return the result in
	npy_intp outdims[2];
    outdims[0] = width;
    outdims[1] = height;
	PyObject *result = PyArray_SimpleNew(2, outdims, NPY_INT);

    int *data_ptr = NULL;
	//fill the output array with the results
	for(int i = 0; i < width*height; i++)
	{
		data_ptr = (int*)PyArray_GETPTR2(result, i % width, i / width);
		*data_ptr = labels[i];
	}

	return result;
};

/*
Parameters(int width, int height, float pairwise_cost, np float array source, np float array sink)
Source and sink parameters should be a 1D numpy array with width*height elements.
Indexing should be done so that source[x+y*width] is the capacity of node located at (x,y)
*/
static PyObject* gridcut_expansion_2D_4C(PyObject* self, PyObject *args,
                                         PyObject *keywds)
{
	PyObject *unary_costs=NULL, *pairwise_costs=NULL, *edges_r=NULL, *edges_d=NULL;
    PyArrayObject *unary_costs_arr=NULL, *pairwise_costs_arr=NULL, *edges_r_arr=NULL, *edges_d_arr=NULL;
	int n_threads=1, block_size=0;


	//parse arguments
	static char *kwlist[] = {"unary_cost", "pairwise_cost",
	                         "edges_right", "edges_down", "n_threads", "block_size", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|ii", kwlist,
	                                 &unary_costs, &pairwise_costs,
	                                 &edges_r, &edges_d, &n_threads, &block_size))
        return NULL;

    unary_costs_arr = (PyArrayObject*)PyArray_FROM_OTF(unary_costs, NPY_FLOAT32, NPY_IN_ARRAY);
    pairwise_costs_arr = (PyArrayObject*)PyArray_FROM_OTF(pairwise_costs, NPY_FLOAT32, NPY_IN_ARRAY);
    edges_r_arr = (PyArrayObject*)PyArray_FROM_OTF(edges_r, NPY_FLOAT32, NPY_IN_ARRAY);
    edges_d_arr = (PyArrayObject*)PyArray_FROM_OTF(edges_d, NPY_FLOAT32, NPY_IN_ARRAY);

    const int width = unary_costs_arr->dimensions[0];
    const int height = unary_costs_arr->dimensions[1];
    const int n_labels = unary_costs_arr->dimensions[2];
    printf("width = %d, height = %d, n_labels = %d", width, height, n_labels);

    if (unary_costs_arr == NULL || pairwise_costs_arr == NULL || edges_r_arr==NULL || edges_d_arr==NULL)
        return NULL;
    float* dataCosts = new float[width * height * n_labels];

    for(int y=0; y<height; y++)
    for(int x=0; x<width; x++)
    {
        for(int label=0;label<n_labels;label++)
        {
            float* data_ptr = (float*)PyArray_GETPTR3(unary_costs_arr, x, y, label);
            dataCosts[(x+y*width)*n_labels+label] = *data_ptr;
            printf("%f", *data_ptr);
        }
    }

    float **pairwise = new float*[n_labels];
    for(int i=0; i<n_labels; i++)
        {
            pairwise[i] = new float[n_labels];
            for(int j=0; j<n_labels; j++)
            {
                float* pw_ptr = (float*)PyArray_GETPTR2(pairwise_costs_arr, i, j);
                pairwise[i][j] = *pw_ptr;
            }
        }

    float** smoothnessCosts = new float*[width * height * 2];

    for(int y=0; y<height; y++)
    for(int x=0; x<width; x++)
    {
        const int xy = x + y * width;

        smoothnessCosts[xy*2+0] = new float[n_labels*n_labels];
        smoothnessCosts[xy*2+1] = new float[n_labels*n_labels];

        float *edge_r, *edge_d;

        if (x<width-1) {
            edge_r = (float*)PyArray_GETPTR2(edges_r_arr, x, y);
        } else {
            *edge_r = 0;
        }

        if (y<height-1) {
            edge_d = (float*)PyArray_GETPTR2(edges_d_arr, x, y);
        } else {
            *edge_d = 0;
        }

        for(int label=0; label<n_labels; label++)
        for(int otherLabel=0; otherLabel<n_labels; otherLabel++)
        {
            const int idx = label + otherLabel * n_labels;
            smoothnessCosts[xy*2+0][idx] = pairwise[label][otherLabel] + *edge_r;
            smoothnessCosts[xy*2+1][idx] = pairwise[label][otherLabel] + *edge_d;
        }
    }

    int* labeling = NULL;

	if(n_threads > 1)
	{
		Expansion_2d_4c_mt* expansion = new Expansion_2d_4c_mt(width, height, n_labels, dataCosts, smoothnessCosts, n_threads, block_size);
        expansion->perform();
        labeling = expansion->get_labeling();
   		delete expansion;
	}
	else
	{
		Expansion_2d_4c* expansion = new Expansion_2d_4c(width, height, n_labels, dataCosts, smoothnessCosts);
        expansion->perform();
        labeling = expansion->get_labeling();
   		delete expansion;
	}

    PyObject *result = labels_to_array(labeling, width, height);

    delete [] dataCosts;
    delete [] smoothnessCosts;
    delete [] labeling;

   	Py_DECREF(unary_costs_arr);
   	Py_DECREF(pairwise_costs_arr);
   	Py_DECREF(edges_r_arr);
   	Py_DECREF(edges_d_arr);

    return result;
};



static PyMethodDef expansion_funcs[] = {

     {"expansion_2D_4C", (PyCFunction)gridcut_expansion_2D_4C,
     METH_VARARGS | METH_KEYWORDS, "a message"},

    {NULL}
};