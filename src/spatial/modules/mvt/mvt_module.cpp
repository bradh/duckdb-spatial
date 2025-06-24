// Mapbox Vector Tiles (MVT) implementation

#include "spatial/modules/mvt/mvt_module.hpp"

#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "spatial/geometry/geometry_serialization.hpp"
#include "spatial/geometry/sgl.hpp"
#include "spatial/spatial_types.hpp"
#include <spatial/util/function_builder.hpp>

namespace duckdb {

namespace {
// ######################################################################################################################
// Util
// ######################################################################################################################

//======================================================================================================================
// LocalState
//======================================================================================================================

class LocalState final : public FunctionLocalState {
public:
    explicit LocalState(ClientContext &context) : arena(BufferAllocator::Get(context)), allocator(arena) {
    }

    static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
                                               FunctionData *bind_data);
    static LocalState &ResetAndGet(ExpressionState &state);

    void Deserialize(const string_t &blob, sgl::geometry &geom);

    string_t Serialize(Vector &vector, const sgl::geometry &geom);

    GeometryAllocator &GetAllocator() {
        return allocator;
    }

private:
    ArenaAllocator arena;
    GeometryAllocator allocator;
};

unique_ptr<FunctionLocalState> LocalState::Init(ExpressionState &state, const BoundFunctionExpression &expr,
                                                FunctionData *bind_data) {
    return make_uniq_base<FunctionLocalState, LocalState>(state.GetContext());
}

LocalState &LocalState::ResetAndGet(ExpressionState &state) {
    auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<LocalState>();
    local_state.arena.Reset();
    return local_state;
}


void LocalState::Deserialize(const string_t &blob, sgl::geometry &geom) {
    Serde::Deserialize(geom, arena, blob.GetDataUnsafe(), blob.GetSize());
}

string_t LocalState::Serialize(Vector &vector, const sgl::geometry &geom) {
    const auto size = Serde::GetRequiredSize(geom);
    auto blob = StringVector::EmptyString(vector, size);
    Serde::Serialize(geom, blob.GetDataWriteable(), size);
    blob.Finalize();
    return blob;
}

} // namespace

namespace {

struct ST_TileEnvelope {
    static constexpr double RADIUS = 6378137.0;
    static constexpr double PI = 3.141592653589793;
    static constexpr double CIRCUMFERENCE = 2 * PI * RADIUS;

    static void ExecuteWebMercator(DataChunk &args, ExpressionState &state, Vector &result) {
        auto &lstate = LocalState::ResetAndGet(state);

        TernaryExecutor::Execute<int32_t, int32_t, int32_t, string_t>(
            args.data[0], args.data[1], args.data[2], result, args.size(),
            [&](int32_t tile_zoom, int32_t tile_x, int32_t tile_y) {
                validate_tile_zoom_argument(tile_zoom);
                uint32_t zoom_extent = 1u << tile_zoom;
                validate_tile_index_arguments(zoom_extent, tile_x, tile_y);
                sgl::geometry bbox;
                get_tile_bbox(lstate.GetAllocator(), zoom_extent, tile_x, tile_y, bbox);
                return lstate.Serialize(result, bbox);
            });
    }

    static void validate_tile_zoom_argument(int32_t tile_zoom) {
        if ((tile_zoom < 0) || (tile_zoom > 30)) {
            throw InvalidInputException("ST_TileEnvelope: tile_zoom must be in the range [0,30]");
        }
    }

    static void validate_tile_index_arguments(uint32_t zoom_extent, int32_t tile_x, int32_t tile_y) {
        if ((tile_x < 0) || ((uint32_t)tile_x >= zoom_extent)) {
            throw InvalidInputException("ST_TileEnvelope: tile_x is out of range for specified tile_zoom");
        }
        if ((tile_y < 0) || ((uint32_t)tile_y >= zoom_extent)) {
            throw InvalidInputException("ST_TileEnvelope: tile_y is out of range for specified tile_zoom");
        }
    }

    static void get_tile_bbox(GeometryAllocator &allocator, uint32_t zoom_extent,
                              int32_t tile_x, int32_t tile_y, sgl::geometry &bbox) {
        double single_tile_width = CIRCUMFERENCE / zoom_extent;
        double single_tile_height = CIRCUMFERENCE / zoom_extent;
        double tile_left = get_tile_left(tile_x, single_tile_width);
        double tile_right = tile_left + single_tile_width;
        double tile_top = get_tile_top(tile_y, single_tile_height);
        double tile_bottom = tile_top - single_tile_height;

        sgl::polygon::init_from_bbox(allocator, tile_left, tile_bottom, tile_right, tile_top, bbox);
    }

    static double get_tile_left(uint32_t tile_x, double single_tile_width) {
        return -0.5 * CIRCUMFERENCE + (tile_x * single_tile_width);
    }

    static double get_tile_top(uint32_t tile_y, double single_tile_height) {
        return 0.5 * CIRCUMFERENCE - (tile_y * single_tile_height);
    }

    //------------------------------------------------------------------------------------------------------------------
    // Documentation
    //------------------------------------------------------------------------------------------------------------------
    static constexpr auto DESCRIPTION = R"(
        The `ST_TileEnvelope` scalar function generates tile envelope rectangular polygons from specified zoom level and tile indices.

        This is used in MVT generation to select the features corresponding to the tile extent. The envelope is in the Web Mercator
        coordinate reference system (EPSG:3857). The tile pyramid starts at zoom level 0, corresponding to a single tile for the
        world. Each zoom level doubles the number of tiles in each direction, such that zoom level 1 is 2 tiles wide by 2 tiles high,
        zoom level 2 is 4 tiles wide by 4 tiles high, and so on. Tile indices start at `[x=0, y=0]` at the top left, and increase
        down and right. For example, at zoom level 2, the top right tile is `[x=3, y=0]`, the bottom left tile is `[x=0, y=3]`, and
        the bottom right is `[x=3, y=3]`.

        ```sql
        SELECT ST_TileEnvelope(2, 3, 1);
        ```
    )";
    static constexpr auto EXAMPLE = R"(
        SELECT ST_TileEnvelope(2, 3, 1);
        ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │                                         st_tileenvelope(2, 3, 1)                                          │
        │                                                 geometry                                                  │
        ├───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ POLYGON ((1.00188E+07 0, 1.00188E+07 1.00188E+07, 2.00375E+07 1.00188E+07, 2.00375E+07 0, 1.00188E+07 0)) │
        └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    )";

    //------------------------------------------------------------------------------------------------------------------
    // Register
    //------------------------------------------------------------------------------------------------------------------
    static void Register(DatabaseInstance &db) {
        FunctionBuilder::RegisterScalar(db, "ST_TileEnvelope", [](ScalarFunctionBuilder &func) {
            func.AddVariant([](ScalarFunctionVariantBuilder &variant) {
                variant.AddParameter("tile_zoom", LogicalType::INTEGER);
                variant.AddParameter("tile_x", LogicalType::INTEGER);
                variant.AddParameter("tile_y", LogicalType::INTEGER);
                variant.SetReturnType(GeoTypes::GEOMETRY());
                variant.SetInit(LocalState::Init);
                variant.SetFunction(ExecuteWebMercator);
            });

            func.SetDescription(DESCRIPTION);
            func.SetExample(EXAMPLE);

            func.SetTag("ext", "spatial");
            func.SetTag("category", "conversion");
        });
    }
};

struct ST_AsMVTGeom{
    static void Execute(DataChunk &args, ExpressionState &state, Vector &result) {
        auto &lstate = LocalState::ResetAndGet(state);

        BinaryExecutor::ExecuteWithNulls<string_t, string_t, string_t>(
            args.data[0], args.data[1], result, args.size(),
            [&](const string_t &geom, const string_t &bounds, ValidityMask &mask, const idx_t row_idx) {
                sgl::geometry sgl_geom;
                lstate.Deserialize(geom, sgl_geom);
                sgl::geometry sgl_bounds;
                lstate.Deserialize(bounds, sgl_bounds);
                // TODO: sanity check bounds
                auto bounds_bbox = sgl::extent_xy::smallest();
                if (sgl::ops::get_total_extent_xy(sgl_bounds, bounds_bbox) != 5) {
                    throw InvalidInputException("ST_AsMVTGeom: invalid bounds geometry");
                }
                double bounds_width = bounds_bbox.max.x - bounds_bbox.min.x;
                double bounds_height = bounds_bbox.max.y - bounds_bbox.min.y;
                // TODO: check for empty input geometry
                // TODO: there is a method on the geom
                // bool is_empty() const;

                sgl::geometry mvt_geom;
                sgl::vertex_xy input_vertex;
                sgl::vertex_xy* output_vertices = new sgl::vertex_xy[sgl_geom.get_vertex_count()]; 
                switch (sgl_geom.get_type()) {
                    case sgl::geometry_type::POINT:
                        input_vertex = sgl_geom.get_vertex_xy(0);
                        output_vertices[0].x = nearbyint((input_vertex.x - bounds_bbox.min.x) * 4096.0 / bounds_width);
                        if (output_vertices[0].x == -0) {
                            output_vertices[0].x = 0;
                        }
                        output_vertices[0].y = nearbyint(4096 - ((input_vertex.y - bounds_bbox.min.y) * 4096.0 / bounds_height));
                        if (output_vertices[0].y == -0) {
                            output_vertices[0].y = 0;
                        }
                        if ((output_vertices[0].x > -256) && (output_vertices[0].x < (4096 + 256)) && (output_vertices[0].y > -256) && (output_vertices[0].y < 4096 + 256)) {
                            mvt_geom.set_type(sgl::geometry_type::POINT);
                            mvt_geom.set_vertex_array(output_vertices, 1);
                        } else {
                            mask.SetInvalid(row_idx);
                            delete[] output_vertices;
                            return string_t {};
                        }
                        break;
                    case sgl::geometry_type::LINESTRING:
                        for (uint32_t i = 0; i < sgl_geom.get_vertex_count(); i++) {
                            input_vertex = sgl_geom.get_vertex_xy(i);
                            output_vertices[i].x = nearbyint((input_vertex.x - bounds_bbox.min.x) * 4096.0 / bounds_width);
                            if (output_vertices[i].x == -0) {
                                output_vertices[i].x = 0;
                            }
                            output_vertices[i].y = nearbyint(4096 - ((input_vertex.y - bounds_bbox.min.y) * 4096.0 / bounds_height));
                            if (output_vertices[i].y == -0) {
                                output_vertices[i].y = 0;
                            }
                            // TODO: clip linestring
                        }
                        mvt_geom.set_type(sgl::geometry_type::LINESTRING);
                        mvt_geom.set_vertex_array(output_vertices, sgl_geom.get_vertex_count());
                        break;
                    default:
                        printf("Unsupported geometry type: %u\n", (uint8_t)sgl_geom.get_type());
                        mask.SetInvalid(row_idx);
                        delete[] output_vertices;
                        return string_t {};
                }

                // TODO: affine transformation into tile coordinate space
                // There is an method for this:
                // void affine_transform(sgl::allocator *alloc, sgl::geometry *geom, const sgl::affine_matrix *matrix);
                // But we can probably do it directly

                // TODO: Snap to integer precision, removing duplicate points

                // TODO: Remove points on straight lines

                // TODO: Remove duplicates in multipoints

                // TODO: check for empty geometry

                // TODO: clip and validate

                // TODO: check for empty geometry
                string_t serialised = lstate.Serialize(result, mvt_geom);
                delete[] output_vertices;
                return serialised;
            });
    }

    //------------------------------------------------------------------------------------------------------------------
    // Documentation
    //------------------------------------------------------------------------------------------------------------------
    static constexpr auto DESCRIPTION = R"(
    )";
    static constexpr auto EXAMPLE = R"(
    )";

    //------------------------------------------------------------------------------------------------------------------
    // Register
    //------------------------------------------------------------------------------------------------------------------
    static void Register(DatabaseInstance &db) {
        FunctionBuilder::RegisterScalar(db, "ST_AsMVTGeom", [](ScalarFunctionBuilder &func) {
            func.AddVariant([](ScalarFunctionVariantBuilder &variant) {
                variant.AddParameter("geom", GeoTypes::GEOMETRY());
                variant.AddParameter("bounds", GeoTypes::GEOMETRY());
                variant.SetReturnType(GeoTypes::GEOMETRY());
                variant.SetInit(LocalState::Init);
                variant.SetFunction(Execute);
            });

            func.SetDescription(DESCRIPTION);
            func.SetExample(EXAMPLE);

            func.SetTag("ext", "spatial");
            func.SetTag("category", "conversion");
        });
    }
};


} // namespace
//------------------------------------------------------------------------------
//  Register
//------------------------------------------------------------------------------
void RegisterMapboxVectorTileModule(DatabaseInstance &db) {
    ST_TileEnvelope::Register(db);
    ST_AsMVTGeom::Register(db);
};

} // namespace duckdb