#ifndef STELLA_VSLAM_DENSE_MODULE_H
#define STELLA_VSLAM_DENSE_MODULE_H

#include "stella_vslam/config.h"
#include "stella_vslam/camera/base.h"

#include <mutex>
#include <atomic>
#include <memory>
#include <future>

#include <yaml-cpp/yaml.h>

namespace dense {
    class PatchMatch;
} // namespace dense

namespace stella_vslam {

using patch_match_module = dense::PatchMatch;

namespace camera {
class base;
} // namespace camera

namespace data {
class keyframe;
class map_database;
} // namespace data

namespace data {
class keyframe;
class map_database;
} // namespace data

class dense_module {
public:
    //! Constructor
    dense_module(const YAML::Node& yaml_node, camera::base* camera, data::map_database* map_db);

    //! Destructor
    ~dense_module();

    //-----------------------------------------
    // main process

    //! Run main loop of the dense module
    void run();

    //! Queue a keyframe to process the dense
    std::shared_future<void> async_add_keyframe(const std::shared_ptr<data::keyframe>& keyfrm);

    //! Check if keyframe is queued
    bool keyframe_is_queued() const;

    //! Get the number of queued keyframes
    unsigned int get_num_queued_keyframes() const;

    //! Check if the dense module is available
    bool is_available() const;

    //-----------------------------------------
    // management for reset process

    //! Request to reset the dense module
    std::shared_future<void> async_reset();

    //-----------------------------------------
    // management for pause process

    //! Request to pause the dense module
    std::shared_future<void> async_pause();

    //! Check if the dense module is requested to be paused
    bool pause_is_requested() const;

    //! Check if the dense module is paused
    bool is_paused() const;

    //! Resume the dense module
    void resume();

    //-----------------------------------------
    // management for terminate process

    //! Request to terminate the dense module
    std::shared_future<void> async_terminate();

    //! Check if the dense module is terminated
    bool is_terminated() const;

private:
    //-----------------------------------------
    // main process

    //! Preprocess and dispatch the new keyframe into the patch match pipeline
    void dispatch_new_keyframe();

    //! Retrieve the dense calculations of the latest keyframe and add it to the map database
    void retrieve_latest_keyframe();

    //-----------------------------------------
    // management for reset process

    //! mutex for access to reset procedure
    mutable std::mutex mtx_reset_;

    //! promise for reset
    std::promise<void> promise_reset_;

    //! future for reset
    std::shared_future<void> future_reset_;

    //! Check and execute reset
    bool reset_is_requested() const;

    //! Reset the variables
    void reset();

    //! flag which indicates whether reset is requested
    bool reset_is_requested_ = false;

    //-----------------------------------------
    // management for pause process

    //! mutex for access to pause procedure
    mutable std::mutex mtx_pause_;

    //! promise for pause
    std::promise<void> promise_pause_;

    //! future for pause
    std::shared_future<void> future_pause_;

    //! Pause the dense module
    void pause();

    //! flag which indicates termination is requested
    bool pause_is_requested_ = false;
    //! flag which indicates whether the main loop is paused
    bool is_paused_ = false;

    //-----------------------------------------
    // management for terminate process

    //! mutex for access to terminate procedure
    mutable std::mutex mtx_terminate_;

    //! promise for terminate
    std::promise<void> promise_terminate_;

    //! future for terminate
    std::shared_future<void> future_terminate_;

    //! Check if termination is requested
    bool terminate_is_requested() const;

    //! Raise the flag which indicates the main loop has been already terminated
    void terminate();

    //! flag which indicates termination is requested
    bool terminate_is_requested_ = false;
    //! flag which indicates whether the main loop is terminated
    bool is_terminated_ = true;

    //-----------------------------------------
    // modules

    //! patch match module
    patch_match_module* patch_match_ = nullptr;

    //-----------------------------------------
    // database

    //! map database
    data::map_database* map_db_ = nullptr;

    //-----------------------------------------
    // keyframe queue

    //! mutex for access to keyframe queue
    mutable std::mutex mtx_keyfrm_queue_;

    //! input queue for keyframe tuples (keyframe, image, mask)
    std::list<std::shared_ptr<data::keyframe>> keyfrms_queue_;

    //! queue for promises
    std::list<std::promise<void>> promise_add_keyfrm_queue_;

    //! processing queue for keyframes
    std::list<std::shared_ptr<data::keyframe>> proc_queue_;

    //-----------------------------------------
    // configurations

    //! camera model
    camera::base* camera_;

    //! image resolution for depth calculation
    cv::Size depth_res_;

    //! number of remaining frames needed to initialize the pipeline
    int count_init_frames_;

};

} // namespace stella_vslam

#endif // STELLA_VSLAM_DENSE_MODULE_H
