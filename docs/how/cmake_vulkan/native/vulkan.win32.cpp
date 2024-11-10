#define VULKAN_HPP_NO_SMART_HANDLE 1
#define VULKAN_HPP_NO_EXCEPTIONS 1
#define VULKAN_HPP_TYPESAFE_CONVERSION 1
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vk_platform.h>

#include <iostream>
#include <functional>
#include <string_view>

#define FRAME_LAG 2

#ifndef NDEBUG
#define VERIFY(x) assert(x)
#else
#define VERIFY(x) ((void)(x))
#endif

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

static char const *const tex_files[] = {"lunarg.ppm"};

#include "linmath.h"
#include "lunarg.ppm.h"

struct vkcube_vs_uniform {
    // Must start with MVP
    float mvp[4][4];
    float position[12 * 3][4];
    float color[12 * 3][4];
};

struct vktexcube_vs_uniform {
    // Must start with MVP
    float mvp[4][4];
    float position[12 * 3][4];
    float attr[12 * 3][4];
};

//--------------------------------------------------------------------------------------
// Mesh and VertexFormat Data
//--------------------------------------------------------------------------------------
// clang-format off
static const float g_vertex_buffer_data[] = {
    -1.0f,-1.0f,-1.0f,  // -X side
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,

    -1.0f,-1.0f,-1.0f,  // -Z side
     1.0f, 1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,

    -1.0f,-1.0f,-1.0f,  // -Y side
     1.0f,-1.0f,-1.0f,
     1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
     1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,

    -1.0f, 1.0f,-1.0f,  // +Y side
    -1.0f, 1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f, 1.0f,-1.0f,

     1.0f, 1.0f,-1.0f,  // +X side
     1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,

    -1.0f, 1.0f, 1.0f,  // +Z side
    -1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
};

static const float g_uv_buffer_data[] = {
    0.0f, 1.0f,  // -X side
    1.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 1.0f,

    1.0f, 1.0f,  // -Z side
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 0.0f,

    1.0f, 0.0f,  // -Y side
    1.0f, 1.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    0.0f, 0.0f,

    1.0f, 0.0f,  // +Y side
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,

    1.0f, 0.0f,  // +X side
    0.0f, 0.0f,
    0.0f, 1.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,

    0.0f, 0.0f,  // +Z side
    0.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
};

typedef struct {
    vk::Image image;
    vk::CommandBuffer cmd;
    vk::CommandBuffer graphics_to_present_cmd;
    vk::ImageView view;
    vk::Buffer uniform_buffer;
    vk::DeviceMemory uniform_memory;
    void *uniform_memory_ptr;
    vk::Framebuffer framebuffer;
    vk::DescriptorSet descriptor_set;
} SwapchainImageResources;

struct texture_object {
    vk::Sampler sampler;

    vk::Image image;
    vk::Buffer buffer;
    vk::ImageLayout imageLayout{vk::ImageLayout::eUndefined};

    vk::MemoryAllocateInfo mem_alloc;
    vk::DeviceMemory mem;
    vk::ImageView view;

    int32_t tex_width{0};
    int32_t tex_height{0};
};

struct VulkanApi {
public:    
    vk::Instance inst;
    vk::PhysicalDevice gpu;
    vk::SurfaceKHR surface;
    vk::Device device;

    std::vector<std::string> enabled_instance_extensions;
    std::vector<std::string> enabled_device_extensions;
    std::vector<std::string> enabled_layers;
    vk::PhysicalDeviceProperties gpu_props;
    std::vector<vk::QueueFamilyProperties> queue_props;    
    vk::PhysicalDeviceFeatures physDevFeatures;
    uint32_t graphics_queue_family_index;
    uint32_t present_queue_family_index;
    vk::Queue graphics_queue;
    vk::Queue present_queue;

    bool separate_present_queue;
    bool use_staging_buffer;

    uint32_t width;
    uint32_t height;    
    vk::Format format;
    vk::ColorSpaceKHR color_space;

    uint32_t frame_index;
    vk::Fence fences[FRAME_LAG];
    vk::Semaphore image_acquired_semaphores[FRAME_LAG];
    vk::Semaphore draw_complete_semaphores[FRAME_LAG];
    vk::Semaphore image_ownership_semaphores[FRAME_LAG];
    vk::PhysicalDeviceMemoryProperties memory_properties;

    vk::CommandPool cmd_pool;
    vk::CommandPool present_cmd_pool;

    vk::CommandBuffer cmd;  // Buffer for initialization commands
    vk::CommandBufferAllocateInfo cmd_alloc_info;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorSetLayout desc_layout;
    vk::PipelineCache pipelineCache;
    vk::RenderPass render_pass;
    vk::Pipeline pipeline;

    uint32_t swapchainImageCount;
    vk::SwapchainKHR swapchain;
    std::unique_ptr<SwapchainImageResources[]> swapchain_image_resources;
    vk::PresentModeKHR presentMode;

    vk::ShaderModule vert_shader_module;
    vk::ShaderModule frag_shader_module;

    vk::DescriptorPool desc_pool;
    vk::DescriptorSet desc_set;

    struct {
        vk::Format format;
        vk::Image image;
        vk::MemoryAllocateInfo mem_alloc;
        vk::DeviceMemory mem;
        vk::ImageView view;
    } depth;

    static int32_t const texture_count = 1;
    texture_object textures[texture_count];
    texture_object staging_texture;

    struct {
        vk::Buffer buf;
        vk::MemoryAllocateInfo mem_alloc;
        vk::DeviceMemory mem;
        vk::DescriptorBufferInfo buffer_info;
    } uniform_data;

    uint32_t current_buffer;
    
    mat4x4 projection_matrix;
    mat4x4 view_matrix;
    mat4x4 model_matrix;

    float spin_angle;
    float spin_increment;
    bool pause;
    bool prepared;

    std::function<void(vk::Instance& inst, vk::SurfaceKHR& surface)> on_create_surface;

    VulkanApi() :
        prepared{false},
        use_staging_buffer{false},
        graphics_queue_family_index{0},
        present_queue_family_index{0},
        width{0},
        height{0},
        swapchainImageCount{0},
        presentMode{vk::PresentModeKHR::eFifo},
        frame_index{0},
        spin_angle{0.0f},
        spin_increment{0.0f},
        pause{false},
        current_buffer{0},
        projection_matrix{},
        view_matrix{},
        model_matrix{} {

        vec3 eye = {0.0f, 3.0f, 5.0f};
        vec3 origin = {0, 0, 0};
        vec3 up = {0.0f, 1.0f, 0.0};

        width = 500;
        height = 500;

        spin_angle = 4.0f;
        spin_increment = 0.2f;
        pause = false;

        mat4x4_perspective(projection_matrix, (float)degreesToRadians(45.0f), 1.0f, 0.1f, 100.0f);
        mat4x4_look_at(view_matrix, eye, origin, up);
        mat4x4_identity(model_matrix);

        projection_matrix[1][1] *= -1;  // Flip projection matrix from GL to Vulkan orientation.            
    }

    auto initialize() {
        return validate() 
            && load_surface_extentions()
            && create_instance()
            && load_swapchain_extention()
            && load_gpu_and_queue_properties();
    }
   
#ifdef WIN32   
    void create_surface_win32(HINSTANCE hInstance, HWND hwnd) {
        auto const createInfo = vk::Win32SurfaceCreateInfoKHR().setHinstance(hInstance).setHwnd(hwnd);
        auto result = inst.createWin32SurfaceKHR(&createInfo, nullptr, &surface);
        VERIFY(result == vk::Result::eSuccess);
    }
#endif    

    bool initialize_swapchain() {
        return create_surface() 
            && set_graphics_and_present_family_indexes()
            && create_device()
            && get_queue_from_device()
            && load_list_of_vk_formats()
            && create_semaphores()
            && get_memory_properties();
    }

    bool prepare() {
        prepared = false;
        prepared = prepare_command_pool()
            && prepare_buffers()
            && prepare_depth()
            && prepare_textures()
            && prepare_cube_data_buffers()
            && prepare_descriptor_layout()
            && prepare_render_pass()
            && prepare_pipeline()
            && allocate_command_buffers()
            && prepare_descriptor_pool()
            && prepare_descriptor_set()
            && prepare_framebuffers()
            && draw_build_cmd()
            && flush_init_cmd()
            && destroy_textures();
        return prepared;
    }

    void resize() {
        // Don't react to resize until after first initialization.
        if (!prepared) {
            return;
        }

        cleanup(false);

        // Second, re-perform the prepare() function, which will re-create the
        // swapchain.
        prepare();
    }    

    void run() {
        draw();
    }

    bool cleanup(bool forExit = true) {
        prepared = false;
        auto result = device.waitIdle();
        VERIFY(result == vk::Result::eSuccess);

        if (forExit) {
            // Wait for fences from present operations
            for (uint32_t i = 0; i < FRAME_LAG; i++) {
                device.waitForFences(1, &fences[i], VK_TRUE, UINT64_MAX);
                device.destroyFence(fences[i], nullptr);
                device.destroySemaphore(image_acquired_semaphores[i], nullptr);
                device.destroySemaphore(draw_complete_semaphores[i], nullptr);
                if (separate_present_queue) {
                    device.destroySemaphore(image_ownership_semaphores[i], nullptr);
                }
            }
        }

        for (uint32_t i = 0; i < swapchainImageCount; i++) {
            device.destroyFramebuffer(swapchain_image_resources[i].framebuffer, nullptr);
        }

        device.destroyDescriptorPool(desc_pool, nullptr);

        device.destroyPipeline(pipeline, nullptr);
        device.destroyPipelineCache(pipelineCache, nullptr);
        device.destroyRenderPass(render_pass, nullptr);
        device.destroyPipelineLayout(pipeline_layout, nullptr);
        device.destroyDescriptorSetLayout(desc_layout, nullptr);

        for (uint32_t i = 0; i < texture_count; i++) {
            device.destroyImageView(textures[i].view, nullptr);
            device.destroyImage(textures[i].image, nullptr);
            device.freeMemory(textures[i].mem, nullptr);
            device.destroySampler(textures[i].sampler, nullptr);
        }

        if (forExit) {        
            device.destroySwapchainKHR(swapchain, nullptr);
        }

        device.destroyImageView(depth.view, nullptr);
        device.destroyImage(depth.image, nullptr);
        device.freeMemory(depth.mem, nullptr);

        for (uint32_t i = 0; i < swapchainImageCount; i++) {
            device.destroyImageView(swapchain_image_resources[i].view, nullptr);
            device.freeCommandBuffers(cmd_pool, 1, &swapchain_image_resources[i].cmd);
            device.destroyBuffer(swapchain_image_resources[i].uniform_buffer, nullptr);
            device.unmapMemory(swapchain_image_resources[i].uniform_memory);
            device.freeMemory(swapchain_image_resources[i].uniform_memory, nullptr);
        }

        device.destroyCommandPool(cmd_pool, nullptr);
        if (separate_present_queue) {
            device.destroyCommandPool(present_cmd_pool, nullptr);
        }

        if (forExit) {    
            device.waitIdle();
            device.destroy(nullptr);
            inst.destroySurfaceKHR(surface, nullptr);

            inst.destroy(nullptr);        
        }

        return true;
    }

private:
    auto to_const_strings(const std::vector<std::string>& vectorOfStrings) {
        auto count = 0;
        auto pnames = std::make_unique<const char*[]>(vectorOfStrings.size());
        std::for_each(vectorOfStrings.begin(), vectorOfStrings.end(), [&] (auto& item) {
            pnames[count++] = item.c_str();
        });    

        return pnames;    
    }

    bool create_surface() {
        if (on_create_surface) {
            on_create_surface(inst, surface);
        }
        
        return true;
    }

    bool validate() {
        uint32_t instance_layer_count = 0;
        std::array<std::string_view, 1> instance_validation_layers = {"VK_LAYER_KHRONOS_validation"};

        // Look for validation layers
        auto validation_found = false;
        auto result = vk::enumerateInstanceLayerProperties(&instance_layer_count, static_cast<vk::LayerProperties *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        if (instance_layer_count > 0) {
            std::unique_ptr<vk::LayerProperties[]> instance_layers(new vk::LayerProperties[instance_layer_count]);
            result = vk::enumerateInstanceLayerProperties(&instance_layer_count, instance_layers.get());
            VERIFY(result == vk::Result::eSuccess);

            std::for_each_n(instance_layers.get(), instance_layer_count, [&] (auto& instance_layer) {
                auto found = std::find_if(instance_validation_layers.begin(), instance_validation_layers.end(), [&] (auto& instance_validation_layer) {
                    return instance_layer.layerName == instance_validation_layer;
                });

                if (found != instance_validation_layers.end()) {
                    enabled_layers.push_back(instance_layer.layerName);
                }
            });
        }

        if (!enabled_layers.size()) {
            error(
                "vkEnumerateInstanceLayerProperties failed to find required validation layer.\n\n"
                "Please look at the Getting Started guide for additional information.\n",
                "vkCreateInstance Failure");
        }

        return true;
    }

    bool load_surface_extentions() {
        /* Look for instance extensions */
        auto surfaceExtFound = false;
        auto platformSurfaceExtFound = false;

        uint32_t instance_extension_count = 0;
        auto result = vk::enumerateInstanceExtensionProperties(nullptr, &instance_extension_count, static_cast<vk::ExtensionProperties *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        if (instance_extension_count > 0) {
            auto instance_extensions = std::make_unique<vk::ExtensionProperties[]>(instance_extension_count);
            result = vk::enumerateInstanceExtensionProperties(nullptr, &instance_extension_count, instance_extensions.get());
            VERIFY(result == vk::Result::eSuccess);

            std::for_each_n(instance_extensions.get(), instance_extension_count, [&] (auto& instance_extension) {
                if (std::string_view(VK_KHR_SURFACE_EXTENSION_NAME) == std::string_view(instance_extension.extensionName)) {
                    surfaceExtFound = true;
                    enabled_instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

                }
#if defined(VK_USE_PLATFORM_WIN32_KHR)
                if (std::string_view(VK_KHR_SURFACE_EXTENSION_NAME) == std::string_view(instance_extension.extensionName) ) {
                    platformSurfaceExtFound = true;
                    enabled_instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
                }
#endif
            });
        }

        if (!surfaceExtFound) {
            error("vkEnumerateInstanceExtensionProperties failed to find the " VK_KHR_SURFACE_EXTENSION_NAME
                    " extension.\n\n"
                    "Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
                    "Please look at the Getting Started guide for additional information.\n",
                    "vkCreateInstance Failure");
        }

        if (!platformSurfaceExtFound) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
            error("vkEnumerateInstanceExtensionProperties failed to find the " VK_KHR_WIN32_SURFACE_EXTENSION_NAME
                    " extension.\n\n"
                    "Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
                    "Please look at the Getting Started guide for additional information.\n",
                    "vkCreateInstance Failure");
#endif
        }        

        return surfaceExtFound || platformSurfaceExtFound;
    }

    bool create_instance() {
        auto const app = vk::ApplicationInfo()
                            .setPApplicationName("App")
                            .setApplicationVersion(0)
                            .setPEngineName("App")
                            .setEngineVersion(0)
                            .setApiVersion(VK_API_VERSION_1_0);

        auto enabled_layers_pnames = to_const_strings(enabled_layers);
        auto enabled_extensions_pnames = to_const_strings(enabled_instance_extensions);

        auto const inst_info = vk::InstanceCreateInfo()
                                .setPApplicationInfo(&app)
                                .setEnabledLayerCount(static_cast<uint32_t>(enabled_layers.size()))
                                .setPpEnabledLayerNames(enabled_layers_pnames.get())
                                .setEnabledExtensionCount(static_cast<uint32_t>(enabled_instance_extensions.size()))
                                .setPpEnabledExtensionNames(enabled_extensions_pnames.get());

        auto result = vk::createInstance(&inst_info, nullptr, &inst);

        if (result == vk::Result::eErrorIncompatibleDriver) {
            error(
                "Cannot find a compatible Vulkan installable client driver (ICD).\n\n"
                "Please look at the Getting Started guide for additional information.\n",
                "vkCreateInstance Failure");
        } else if (result == vk::Result::eErrorExtensionNotPresent) {
            error(
                "Cannot find a specified extension library.\n"
                "Make sure your layers path is set appropriately.\n",
                "vkCreateInstance Failure");
        } else if (result != vk::Result::eSuccess) {
            error(
                "vkCreateInstance failed.\n\n"
                "Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
                "Please look at the Getting Started guide for additional information.\n",
                "vkCreateInstance Failure");
        }        

        return true;
    }

    bool load_swapchain_extention() {
        /* Look for device extensions */
        uint32_t device_extension_count = 0;
        bool swapchainExtFound = false;

        /* Make initial call to query gpu_count, then second call for gpu info*/
        uint32_t gpu_count;
        auto result = inst.enumeratePhysicalDevices(&gpu_count, static_cast<vk::PhysicalDevice *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        if (gpu_count > 0) {
            auto physical_devices = std::make_unique<vk::PhysicalDevice[]>(gpu_count);
            result = inst.enumeratePhysicalDevices(&gpu_count, physical_devices.get());
            VERIFY(result == vk::Result::eSuccess);
            /* For this app we just grab the first physical device */
            gpu = physical_devices[0];
        } else {
            error(
                "vkEnumeratePhysicalDevices reported zero accessible devices.\n\n"
                "Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
                "Please look at the Getting Started guide for additional information.\n",
                "vkEnumeratePhysicalDevices Failure");
        }        

        result = gpu.enumerateDeviceExtensionProperties(nullptr, &device_extension_count, static_cast<vk::ExtensionProperties *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        if (device_extension_count > 0) {
            auto device_extensions = std::make_unique<vk::ExtensionProperties[]>(device_extension_count);
            result = gpu.enumerateDeviceExtensionProperties(nullptr, &device_extension_count, device_extensions.get());
            VERIFY(result == vk::Result::eSuccess);

            std::for_each_n(device_extensions.get(), device_extension_count, [&] (auto& device_extension) {
                if (std::string_view(VK_KHR_SWAPCHAIN_EXTENSION_NAME) == std::string_view(device_extension.extensionName)) {
                    swapchainExtFound = true;
                    enabled_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
                }
            });
        }

        if (!swapchainExtFound) {
            error("vkEnumerateDeviceExtensionProperties failed to find the " VK_KHR_SWAPCHAIN_EXTENSION_NAME
                    " extension.\n\n"
                    "Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
                    "Please look at the Getting Started guide for additional information.\n",
                    "vkCreateInstance Failure");
        }

        return swapchainExtFound;
    }

    bool load_gpu_and_queue_properties() {
        gpu.getProperties(&gpu_props);

        /* Call with nullptr data to get count */
        uint32_t queue_family_count = 0;
        gpu.getQueueFamilyProperties(&queue_family_count, static_cast<vk::QueueFamilyProperties *>(nullptr));
        VERIFY(queue_family_count >= 1);

        auto queue_props_ptrs = std::make_unique<vk::QueueFamilyProperties[]>(queue_family_count);
        gpu.getQueueFamilyProperties(&queue_family_count, queue_props_ptrs.get());

        std::for_each_n(queue_props_ptrs.get(), queue_family_count, [&] (auto& queue_prop) {
            queue_props.push_back(queue_prop);
        });

        // Query fine-grained feature support for this device.
        //  If app has specific feature requirements it should check supported
        //  features based on this query
        gpu.getFeatures(&physDevFeatures);        
        return true;
    }

    bool set_graphics_and_present_family_indexes() {
        // Iterate over each queue to learn whether it supports presenting:
        auto queue_family_count = queue_props.size();
        auto supportsPresent = std::make_unique<vk::Bool32[]>(queue_family_count);
        for (auto i = 0; i < queue_family_count; i++) {
            gpu.getSurfaceSupportKHR(i, surface, &supportsPresent[i]);
        }

        auto graphicsQueueFamilyIndex = UINT32_MAX;
        auto presentQueueFamilyIndex = UINT32_MAX;
        for (auto i = 0; i < queue_family_count; i++) {
            if (queue_props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                if (graphicsQueueFamilyIndex == UINT32_MAX) {
                    graphicsQueueFamilyIndex = i;
                }

                if (supportsPresent[i] == VK_TRUE) {
                    graphicsQueueFamilyIndex = i;
                    presentQueueFamilyIndex = i;
                    break;
                }
            }
        }

        if (presentQueueFamilyIndex == UINT32_MAX) {
            // If didn't find a queue that supports both graphics and present,
            // then
            // find a separate present queue.
            for (auto i = 0; i < queue_family_count; ++i) {
                if (supportsPresent[i] == VK_TRUE) {
                    presentQueueFamilyIndex = i;
                    break;
                }
            }
        }

        // Generate error if could not find both a graphics and a present queue
        if (graphicsQueueFamilyIndex == UINT32_MAX || presentQueueFamilyIndex == UINT32_MAX) {
            error("Could not find both graphics and present queues\n", "Swapchain Initialization Failure");
        }

        graphics_queue_family_index = graphicsQueueFamilyIndex;
        present_queue_family_index = presentQueueFamilyIndex;
        separate_present_queue = (graphics_queue_family_index != present_queue_family_index);        

        return true;
    }

    bool create_device() {
        float const priorities[1] = {0.0};

        vk::DeviceQueueCreateInfo queues[2];
        queues[0].setQueueFamilyIndex(graphics_queue_family_index);
        queues[0].setQueueCount(1);
        queues[0].setPQueuePriorities(priorities);

        auto enabled_extensions_pnames = to_const_strings(enabled_device_extensions);

        auto deviceInfo = vk::DeviceCreateInfo()
                            .setQueueCreateInfoCount(1)
                            .setPQueueCreateInfos(queues)
                            .setEnabledLayerCount(0)
                            .setPpEnabledLayerNames(nullptr)
                            .setEnabledExtensionCount(static_cast<uint32_t>(enabled_device_extensions.size()))
                            .setPpEnabledExtensionNames(enabled_extensions_pnames.get())
                            .setPEnabledFeatures(nullptr);

        if (separate_present_queue) {
            queues[1].setQueueFamilyIndex(present_queue_family_index);
            queues[1].setQueueCount(1);
            queues[1].setPQueuePriorities(priorities);
            deviceInfo.setQueueCreateInfoCount(2);
        }

        auto result = gpu.createDevice(&deviceInfo, nullptr, &device);
        VERIFY(result == vk::Result::eSuccess);        

        return true;
    }

    bool get_queue_from_device() {
        device.getQueue(graphics_queue_family_index, 0, &graphics_queue);
        if (!separate_present_queue) {
            present_queue = graphics_queue;
        } else {
            device.getQueue(present_queue_family_index, 0, &present_queue);
        }

        return true;
    }
    
    bool load_list_of_vk_formats() {
        uint32_t formatCount;
        auto result = gpu.getSurfaceFormatsKHR(surface, &formatCount, static_cast<vk::SurfaceFormatKHR *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        std::unique_ptr<vk::SurfaceFormatKHR[]> surfFormats(new vk::SurfaceFormatKHR[formatCount]);
        result = gpu.getSurfaceFormatsKHR(surface, &formatCount, surfFormats.get());
        VERIFY(result == vk::Result::eSuccess);

        // If the format list includes just one entry of VK_FORMAT_UNDEFINED,
        // the surface has no preferred format.  Otherwise, at least one
        // supported format will be returned.
        if (formatCount == 1 && surfFormats[0].format == vk::Format::eUndefined) {
            format = vk::Format::eB8G8R8A8Unorm;
        } else {
            assert(formatCount >= 1);
            format = surfFormats[0].format;
        }

        color_space = surfFormats[0].colorSpace;
        return true;
    }

    bool create_semaphores() {
        // Create semaphores to synchronize acquiring presentable buffers before
        // rendering and waiting for drawing to be complete before presenting
        auto const semaphoreCreateInfo = vk::SemaphoreCreateInfo();

        // Create fences that we can use to throttle if we get too far
        // ahead of the image presents
        auto const fence_ci = vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled);
        for (uint32_t i = 0; i < FRAME_LAG; i++) {
            auto result = device.createFence(&fence_ci, nullptr, &fences[i]);
            VERIFY(result == vk::Result::eSuccess);

            result = device.createSemaphore(&semaphoreCreateInfo, nullptr, &image_acquired_semaphores[i]);
            VERIFY(result == vk::Result::eSuccess);

            result = device.createSemaphore(&semaphoreCreateInfo, nullptr, &draw_complete_semaphores[i]);
            VERIFY(result == vk::Result::eSuccess);

            if (separate_present_queue) {
                result = device.createSemaphore(&semaphoreCreateInfo, nullptr, &image_ownership_semaphores[i]);
                VERIFY(result == vk::Result::eSuccess);
            }
        }        

        return true;
    }

    bool get_memory_properties() {
        // Get Memory information and properties
        gpu.getMemoryProperties(&memory_properties);

        return true;
    }

    bool prepare_command_pool() {
        auto const cmd_pool_info = vk::CommandPoolCreateInfo().setQueueFamilyIndex(graphics_queue_family_index);
        auto result = device.createCommandPool(&cmd_pool_info, nullptr, &cmd_pool);
        VERIFY(result == vk::Result::eSuccess);

        cmd_alloc_info = vk::CommandBufferAllocateInfo()
                            .setCommandPool(cmd_pool)
                            .setLevel(vk::CommandBufferLevel::ePrimary)
                            .setCommandBufferCount(1);

        result = device.allocateCommandBuffers(&cmd_alloc_info, &cmd);
        VERIFY(result == vk::Result::eSuccess);

        auto const cmd_buf_info = vk::CommandBufferBeginInfo().setPInheritanceInfo(nullptr);

        result = cmd.begin(&cmd_buf_info);
        VERIFY(result == vk::Result::eSuccess);    

        return true;    
    }

    bool prepare_buffers() {
        vk::SwapchainKHR oldSwapchain = swapchain;

        // Check the surface capabilities and formats
        vk::SurfaceCapabilitiesKHR surfCapabilities;
        auto result = gpu.getSurfaceCapabilitiesKHR(surface, &surfCapabilities);
        VERIFY(result == vk::Result::eSuccess);

        uint32_t presentModeCount;
        result = gpu.getSurfacePresentModesKHR(surface, &presentModeCount, static_cast<vk::PresentModeKHR *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        std::unique_ptr<vk::PresentModeKHR[]> presentModes(new vk::PresentModeKHR[presentModeCount]);
        result = gpu.getSurfacePresentModesKHR(surface, &presentModeCount, presentModes.get());
        VERIFY(result == vk::Result::eSuccess);

        vk::Extent2D swapchainExtent;
        // width and height are either both -1, or both not -1.
        if (surfCapabilities.currentExtent.width == (uint32_t)-1) {
            // If the surface size is undefined, the size is set to
            // the size of the images requested.
            swapchainExtent.width = width;
            swapchainExtent.height = height;
        } else {
            // If the surface size is defined, the swap chain size must match
            swapchainExtent = surfCapabilities.currentExtent;
            width = surfCapabilities.currentExtent.width;
            height = surfCapabilities.currentExtent.height;
        }

        // The FIFO present mode is guaranteed by the spec to be supported
        // and to have no tearing.  It's a great default present mode to use.
        vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

        //  There are times when you may wish to use another present mode.  The
        //  following code shows how to select them, and the comments provide some
        //  reasons you may wish to use them.
        //
        // It should be noted that Vulkan 1.0 doesn't provide a method for
        // synchronizing rendering with the presentation engine's display.  There
        // is a method provided for throttling rendering with the display, but
        // there are some presentation engines for which this method will not work.
        // If an application doesn't throttle its rendering, and if it renders much
        // faster than the refresh rate of the display, this can waste power on
        // mobile devices.  That is because power is being spent rendering images
        // that may never be seen.

        // VK_PRESENT_MODE_IMMEDIATE_KHR is for applications that don't care
        // about
        // tearing, or have some way of synchronizing their rendering with the
        // display.
        // VK_PRESENT_MODE_MAILBOX_KHR may be useful for applications that
        // generally render a new presentable image every refresh cycle, but are
        // occasionally early.  In this case, the application wants the new
        // image
        // to be displayed instead of the previously-queued-for-presentation
        // image
        // that has not yet been displayed.
        // VK_PRESENT_MODE_FIFO_RELAXED_KHR is for applications that generally
        // render a new presentable image every refresh cycle, but are
        // occasionally
        // late.  In this case (perhaps because of stuttering/latency concerns),
        // the application wants the late image to be immediately displayed,
        // even
        // though that may mean some tearing.

        if (presentMode != swapchainPresentMode) {
            for (size_t i = 0; i < presentModeCount; ++i) {
                if (presentModes[i] == presentMode) {
                    swapchainPresentMode = presentMode;
                    break;
                }
            }
        }

        if (swapchainPresentMode != presentMode) {
            error("Present mode specified is not supported\n", "Present mode unsupported");
        }

        // Determine the number of VkImages to use in the swap chain.
        // Application desires to acquire 3 images at a time for triple
        // buffering
        uint32_t desiredNumOfSwapchainImages = 3;
        if (desiredNumOfSwapchainImages < surfCapabilities.minImageCount) {
            desiredNumOfSwapchainImages = surfCapabilities.minImageCount;
        }

        // If maxImageCount is 0, we can ask for as many images as we want,
        // otherwise
        // we're limited to maxImageCount
        if ((surfCapabilities.maxImageCount > 0) && (desiredNumOfSwapchainImages > surfCapabilities.maxImageCount)) {
            // Application must settle for fewer images than desired:
            desiredNumOfSwapchainImages = surfCapabilities.maxImageCount;
        }

        vk::SurfaceTransformFlagBitsKHR preTransform;
        if (surfCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
            preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
        } else {
            preTransform = surfCapabilities.currentTransform;
        }

        // Find a supported composite alpha mode - one of these is guaranteed to be set
        vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        vk::CompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
            vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
            vk::CompositeAlphaFlagBitsKHR::eInherit,
        };
        for (uint32_t i = 0; i < ARRAY_SIZE(compositeAlphaFlags); i++) {
            if (surfCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
                compositeAlpha = compositeAlphaFlags[i];
                break;
            }
        }

        auto const swapchain_ci = vk::SwapchainCreateInfoKHR()
                                    .setSurface(surface)
                                    .setMinImageCount(desiredNumOfSwapchainImages)
                                    .setImageFormat(format)
                                    .setImageColorSpace(color_space)
                                    .setImageExtent({swapchainExtent.width, swapchainExtent.height})
                                    .setImageArrayLayers(1)
                                    .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                                    .setImageSharingMode(vk::SharingMode::eExclusive)
                                    .setQueueFamilyIndexCount(0)
                                    .setPQueueFamilyIndices(nullptr)
                                    .setPreTransform(preTransform)
                                    .setCompositeAlpha(compositeAlpha)
                                    .setPresentMode(swapchainPresentMode)
                                    .setClipped(true)
                                    .setOldSwapchain(oldSwapchain);

        result = device.createSwapchainKHR(&swapchain_ci, nullptr, &swapchain);
        VERIFY(result == vk::Result::eSuccess);

        // If we just re-created an existing swapchain, we should destroy the
        // old swapchain at this point.
        // Note: destroying the swapchain also cleans up all its associated
        // presentable images once the platform is done with them.
        if (oldSwapchain) {
            device.destroySwapchainKHR(oldSwapchain, nullptr);
        }

        result = device.getSwapchainImagesKHR(swapchain, &swapchainImageCount, static_cast<vk::Image *>(nullptr));
        VERIFY(result == vk::Result::eSuccess);

        std::unique_ptr<vk::Image[]> swapchainImages(new vk::Image[swapchainImageCount]);
        result = device.getSwapchainImagesKHR(swapchain, &swapchainImageCount, swapchainImages.get());
        VERIFY(result == vk::Result::eSuccess);

        swapchain_image_resources.reset(new SwapchainImageResources[swapchainImageCount]);

        for (uint32_t i = 0; i < swapchainImageCount; ++i) {
            auto color_image_view = vk::ImageViewCreateInfo()
                                        .setViewType(vk::ImageViewType::e2D)
                                        .setFormat(format)
                                        .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

            swapchain_image_resources[i].image = swapchainImages[i];

            color_image_view.image = swapchain_image_resources[i].image;

            result = device.createImageView(&color_image_view, nullptr, &swapchain_image_resources[i].view);
            VERIFY(result == vk::Result::eSuccess);
        }

        return true;
    }

    bool prepare_depth() {
        depth.format = vk::Format::eD16Unorm;

        auto const image = vk::ImageCreateInfo()
                            .setImageType(vk::ImageType::e2D)
                            .setFormat(depth.format)
                            .setExtent({(uint32_t)width, (uint32_t)height, 1})
                            .setMipLevels(1)
                            .setArrayLayers(1)
                            .setSamples(vk::SampleCountFlagBits::e1)
                            .setTiling(vk::ImageTiling::eOptimal)
                            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
                            .setSharingMode(vk::SharingMode::eExclusive)
                            .setQueueFamilyIndexCount(0)
                            .setPQueueFamilyIndices(nullptr)
                            .setInitialLayout(vk::ImageLayout::eUndefined);

        auto result = device.createImage(&image, nullptr, &depth.image);
        VERIFY(result == vk::Result::eSuccess);

        vk::MemoryRequirements mem_reqs;
        device.getImageMemoryRequirements(depth.image, &mem_reqs);

        depth.mem_alloc.setAllocationSize(mem_reqs.size);
        depth.mem_alloc.setMemoryTypeIndex(0);

        auto const pass = memory_type_from_properties(mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                    &depth.mem_alloc.memoryTypeIndex);
        VERIFY(pass);

        result = device.allocateMemory(&depth.mem_alloc, nullptr, &depth.mem);
        VERIFY(result == vk::Result::eSuccess);

        result = device.bindImageMemory(depth.image, depth.mem, 0);
        VERIFY(result == vk::Result::eSuccess);

        auto const view = vk::ImageViewCreateInfo()
                            .setImage(depth.image)
                            .setViewType(vk::ImageViewType::e2D)
                            .setFormat(depth.format)
                            .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
        result = device.createImageView(&view, nullptr, &depth.view);
        VERIFY(result == vk::Result::eSuccess);

        return true;       
    }

    void update_data_buffer() {
        mat4x4 VP;
        mat4x4_mul(VP, projection_matrix, view_matrix);

        // Rotate around the Y axis
        mat4x4 Model;
        mat4x4_dup(Model, model_matrix);
        mat4x4_rotate(model_matrix, Model, 0.0f, 1.0f, 0.0f, (float)degreesToRadians(spin_angle));

        mat4x4 MVP;
        mat4x4_mul(MVP, VP, model_matrix);

        memcpy(swapchain_image_resources[current_buffer].uniform_memory_ptr, (const void *)&MVP[0][0], sizeof(MVP));
    }

    /* Convert ppm image data from header file into RGBA texture image */
    bool loadTexture(const char *filename, uint8_t *rgba_data, vk::SubresourceLayout *layout, int32_t *width, int32_t *height) {
        (void)filename;
        char *cPtr;
        cPtr = (char *)lunarg_ppm;
        if ((unsigned char *)cPtr >= (lunarg_ppm + lunarg_ppm_len) || strncmp(cPtr, "P6\n", 3)) {
            return false;
        }
        while (strncmp(cPtr++, "\n", 1))
            ;
        sscanf(cPtr, "%u %u", width, height);
        if (rgba_data == NULL) {
            return true;
        }
        while (strncmp(cPtr++, "\n", 1))
            ;
        if ((unsigned char *)cPtr >= (lunarg_ppm + lunarg_ppm_len) || strncmp(cPtr, "255\n", 4)) {
            return false;
        }
        while (strncmp(cPtr++, "\n", 1))
            ;
        for (int y = 0; y < *height; y++) {
            uint8_t *rowPtr = rgba_data;
            for (int x = 0; x < *width; x++) {
                memcpy(rowPtr, cPtr, 3);
                rowPtr[3] = 255; /* Alpha of 1 */
                rowPtr += 4;
                cPtr += 3;
            }
            rgba_data += layout->rowPitch;
        }
        return true;
    }

    void prepare_texture_buffer(const char *filename, texture_object *tex_obj) {
        int32_t tex_width;
        int32_t tex_height;

        if (!loadTexture(filename, NULL, NULL, &tex_width, &tex_height)) {
            error("Failed to load textures", "Load Texture Failure");
        }

        tex_obj->tex_width = tex_width;
        tex_obj->tex_height = tex_height;

        auto const buffer_create_info = vk::BufferCreateInfo()
                                            .setSize(tex_width * tex_height * 4)
                                            .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                                            .setSharingMode(vk::SharingMode::eExclusive)
                                            .setQueueFamilyIndexCount(0)
                                            .setPQueueFamilyIndices(nullptr);

        auto result = device.createBuffer(&buffer_create_info, nullptr, &tex_obj->buffer);
        VERIFY(result == vk::Result::eSuccess);

        vk::MemoryRequirements mem_reqs;
        device.getBufferMemoryRequirements(tex_obj->buffer, &mem_reqs);

        tex_obj->mem_alloc.setAllocationSize(mem_reqs.size);
        tex_obj->mem_alloc.setMemoryTypeIndex(0);

        vk::MemoryPropertyFlags requirements = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        auto pass = memory_type_from_properties(mem_reqs.memoryTypeBits, requirements, &tex_obj->mem_alloc.memoryTypeIndex);
        VERIFY(pass == true);

        result = device.allocateMemory(&tex_obj->mem_alloc, nullptr, &(tex_obj->mem));
        VERIFY(result == vk::Result::eSuccess);

        result = device.bindBufferMemory(tex_obj->buffer, tex_obj->mem, 0);
        VERIFY(result == vk::Result::eSuccess);

        vk::SubresourceLayout layout;
        layout.rowPitch = tex_width * 4;
        auto data = device.mapMemory(tex_obj->mem, 0, tex_obj->mem_alloc.allocationSize);
        VERIFY(data.result == vk::Result::eSuccess);

        if (!loadTexture(filename, (uint8_t *)data.value, &layout, &tex_width, &tex_height)) {
            fprintf(stderr, "Error loading texture: %s\n", filename);
        }

        device.unmapMemory(tex_obj->mem);
    }

    void prepare_texture_image(const char *filename, texture_object *tex_obj, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                                    vk::MemoryPropertyFlags required_props) {
        int32_t tex_width;
        int32_t tex_height;
        if (!loadTexture(filename, nullptr, nullptr, &tex_width, &tex_height)) {
            error("Failed to load textures", "Load Texture Failure");
        }

        tex_obj->tex_width = tex_width;
        tex_obj->tex_height = tex_height;

        auto const image_create_info = vk::ImageCreateInfo()
                                        .setImageType(vk::ImageType::e2D)
                                        .setFormat(vk::Format::eR8G8B8A8Unorm)
                                        .setExtent({(uint32_t)tex_width, (uint32_t)tex_height, 1})
                                        .setMipLevels(1)
                                        .setArrayLayers(1)
                                        .setSamples(vk::SampleCountFlagBits::e1)
                                        .setTiling(tiling)
                                        .setUsage(usage)
                                        .setSharingMode(vk::SharingMode::eExclusive)
                                        .setQueueFamilyIndexCount(0)
                                        .setPQueueFamilyIndices(nullptr)
                                        .setInitialLayout(vk::ImageLayout::ePreinitialized);

        auto result = device.createImage(&image_create_info, nullptr, &tex_obj->image);
        VERIFY(result == vk::Result::eSuccess);

        vk::MemoryRequirements mem_reqs;
        device.getImageMemoryRequirements(tex_obj->image, &mem_reqs);

        tex_obj->mem_alloc.setAllocationSize(mem_reqs.size);
        tex_obj->mem_alloc.setMemoryTypeIndex(0);

        auto pass = memory_type_from_properties(mem_reqs.memoryTypeBits, required_props, &tex_obj->mem_alloc.memoryTypeIndex);
        VERIFY(pass == true);

        result = device.allocateMemory(&tex_obj->mem_alloc, nullptr, &(tex_obj->mem));
        VERIFY(result == vk::Result::eSuccess);

        result = device.bindImageMemory(tex_obj->image, tex_obj->mem, 0);
        VERIFY(result == vk::Result::eSuccess);

        if (required_props & vk::MemoryPropertyFlagBits::eHostVisible) {
            auto const subres = vk::ImageSubresource().setAspectMask(vk::ImageAspectFlagBits::eColor).setMipLevel(0).setArrayLayer(0);
            vk::SubresourceLayout layout;
            device.getImageSubresourceLayout(tex_obj->image, &subres, &layout);

            auto data = device.mapMemory(tex_obj->mem, 0, tex_obj->mem_alloc.allocationSize);
            VERIFY(data.result == vk::Result::eSuccess);

            if (!loadTexture(filename, (uint8_t *)data.value, &layout, &tex_width, &tex_height)) {
                fprintf(stderr, "Error loading texture: %s\n", filename);
            }

            device.unmapMemory(tex_obj->mem);
        }

        tex_obj->imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    void set_image_layout(vk::Image image, vk::ImageAspectFlags aspectMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                vk::AccessFlags srcAccessMask, vk::PipelineStageFlags src_stages, vk::PipelineStageFlags dest_stages) {
        assert(cmd);

        auto DstAccessMask = [](vk::ImageLayout const &layout) {
            vk::AccessFlags flags;

            switch (layout) {
                case vk::ImageLayout::eTransferDstOptimal:
                    // Make sure anything that was copying from this image has
                    // completed
                    flags = vk::AccessFlagBits::eTransferWrite;
                    break;
                case vk::ImageLayout::eColorAttachmentOptimal:
                    flags = vk::AccessFlagBits::eColorAttachmentWrite;
                    break;
                case vk::ImageLayout::eDepthStencilAttachmentOptimal:
                    flags = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
                    break;
                case vk::ImageLayout::eShaderReadOnlyOptimal:
                    // Make sure any Copy or CPU writes to image are flushed
                    flags = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eInputAttachmentRead;
                    break;
                case vk::ImageLayout::eTransferSrcOptimal:
                    flags = vk::AccessFlagBits::eTransferRead;
                    break;
                case vk::ImageLayout::ePresentSrcKHR:
                    flags = vk::AccessFlagBits::eMemoryRead;
                    break;
                default:
                    break;
            }

            return flags;
        };

        auto const barrier = vk::ImageMemoryBarrier()
                                .setSrcAccessMask(srcAccessMask)
                                .setDstAccessMask(DstAccessMask(newLayout))
                                .setOldLayout(oldLayout)
                                .setNewLayout(newLayout)
                                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                .setImage(image)
                                .setSubresourceRange(vk::ImageSubresourceRange(aspectMask, 0, 1, 0, 1));

        cmd.pipelineBarrier(src_stages, dest_stages, vk::DependencyFlagBits(), 0, nullptr, 0, nullptr, 1, &barrier);
    }

    bool prepare_textures() {
        vk::Format const tex_format = vk::Format::eR8G8B8A8Unorm;
        vk::FormatProperties props;
        gpu.getFormatProperties(tex_format, &props);

        for (uint32_t i = 0; i < texture_count; i++) {
            if ((props.linearTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage) && !use_staging_buffer) {
                /* Device can texture using linear textures */
                prepare_texture_image(tex_files[i], &textures[i], vk::ImageTiling::eLinear, vk::ImageUsageFlagBits::eSampled,
                                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
                // Nothing in the pipeline needs to be complete to start, and don't allow fragment
                // shader to run until layout transition completes
                set_image_layout(textures[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::ePreinitialized,
                                textures[i].imageLayout, vk::AccessFlagBits(), vk::PipelineStageFlagBits::eTopOfPipe,
                                vk::PipelineStageFlagBits::eFragmentShader);
                staging_texture.image = vk::Image();
            } else if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage) {
                /* Must use staging buffer to copy linear texture to optimized */

                prepare_texture_buffer(tex_files[i], &staging_texture);

                prepare_texture_image(tex_files[i], &textures[i], vk::ImageTiling::eOptimal,
                                    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal);

                set_image_layout(textures[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::ePreinitialized,
                                vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits(), vk::PipelineStageFlagBits::eTopOfPipe,
                                vk::PipelineStageFlagBits::eTransfer);

                auto const subresource = vk::ImageSubresourceLayers()
                                            .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                            .setMipLevel(0)
                                            .setBaseArrayLayer(0)
                                            .setLayerCount(1);

                auto const copy_region =
                    vk::BufferImageCopy()
                        .setBufferOffset(0)
                        .setBufferRowLength(staging_texture.tex_width)
                        .setBufferImageHeight(staging_texture.tex_height)
                        .setImageSubresource(subresource)
                        .setImageOffset({0, 0, 0})
                        .setImageExtent({(uint32_t)staging_texture.tex_width, (uint32_t)staging_texture.tex_height, 1});

                cmd.copyBufferToImage(staging_texture.buffer, textures[i].image, vk::ImageLayout::eTransferDstOptimal, 1, &copy_region);

                set_image_layout(textures[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
                                textures[i].imageLayout, vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eFragmentShader);
            } else {
                assert(!"No support for R8G8B8A8_UNORM as texture image format");
            }

            auto const samplerInfo = vk::SamplerCreateInfo()
                                        .setMagFilter(vk::Filter::eNearest)
                                        .setMinFilter(vk::Filter::eNearest)
                                        .setMipmapMode(vk::SamplerMipmapMode::eNearest)
                                        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
                                        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
                                        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
                                        .setMipLodBias(0.0f)
                                        .setAnisotropyEnable(VK_FALSE)
                                        .setMaxAnisotropy(1)
                                        .setCompareEnable(VK_FALSE)
                                        .setCompareOp(vk::CompareOp::eNever)
                                        .setMinLod(0.0f)
                                        .setMaxLod(0.0f)
                                        .setBorderColor(vk::BorderColor::eFloatOpaqueWhite)
                                        .setUnnormalizedCoordinates(VK_FALSE);

            auto result = device.createSampler(&samplerInfo, nullptr, &textures[i].sampler);
            VERIFY(result == vk::Result::eSuccess);

            auto const viewInfo = vk::ImageViewCreateInfo()
                                    .setImage(textures[i].image)
                                    .setViewType(vk::ImageViewType::e2D)
                                    .setFormat(tex_format)
                                    .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

            result = device.createImageView(&viewInfo, nullptr, &textures[i].view);
            VERIFY(result == vk::Result::eSuccess);
        }

        return true;
    }

    bool prepare_cube_data_buffers() {
        mat4x4 VP;
        mat4x4_mul(VP, projection_matrix, view_matrix);

        mat4x4 MVP;
        mat4x4_mul(MVP, VP, model_matrix);

        vktexcube_vs_uniform data;
        memcpy(data.mvp, MVP, sizeof(MVP));
        //    dumpMatrix("MVP", MVP)

        for (int32_t i = 0; i < 12 * 3; i++) {
            data.position[i][0] = g_vertex_buffer_data[i * 3];
            data.position[i][1] = g_vertex_buffer_data[i * 3 + 1];
            data.position[i][2] = g_vertex_buffer_data[i * 3 + 2];
            data.position[i][3] = 1.0f;
            data.attr[i][0] = g_uv_buffer_data[2 * i];
            data.attr[i][1] = g_uv_buffer_data[2 * i + 1];
            data.attr[i][2] = 0;
            data.attr[i][3] = 0;
        }

        auto const buf_info = vk::BufferCreateInfo().setSize(sizeof(data)).setUsage(vk::BufferUsageFlagBits::eUniformBuffer);

        for (unsigned int i = 0; i < swapchainImageCount; i++) {
            auto result = device.createBuffer(&buf_info, nullptr, &swapchain_image_resources[i].uniform_buffer);
            VERIFY(result == vk::Result::eSuccess);

            vk::MemoryRequirements mem_reqs;
            device.getBufferMemoryRequirements(swapchain_image_resources[i].uniform_buffer, &mem_reqs);

            auto mem_alloc = vk::MemoryAllocateInfo().setAllocationSize(mem_reqs.size).setMemoryTypeIndex(0);

            bool const pass = memory_type_from_properties(
                mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                &mem_alloc.memoryTypeIndex);
            VERIFY(pass);

            result = device.allocateMemory(&mem_alloc, nullptr, &swapchain_image_resources[i].uniform_memory);
            VERIFY(result == vk::Result::eSuccess);

            result = device.mapMemory(swapchain_image_resources[i].uniform_memory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags(),
                                    &swapchain_image_resources[i].uniform_memory_ptr);
            VERIFY(result == vk::Result::eSuccess);

            memcpy(swapchain_image_resources[i].uniform_memory_ptr, &data, sizeof data);

            result =
                device.bindBufferMemory(swapchain_image_resources[i].uniform_buffer, swapchain_image_resources[i].uniform_memory, 0);
            VERIFY(result == vk::Result::eSuccess);
        }

        return true;
    }    

    bool prepare_descriptor_layout() {
        vk::DescriptorSetLayoutBinding const layout_bindings[2] = {vk::DescriptorSetLayoutBinding()
                                                                    .setBinding(0)
                                                                    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                                                                    .setDescriptorCount(1)
                                                                    .setStageFlags(vk::ShaderStageFlagBits::eVertex)
                                                                    .setPImmutableSamplers(nullptr),
                                                                vk::DescriptorSetLayoutBinding()
                                                                    .setBinding(1)
                                                                    .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                                                                    .setDescriptorCount(texture_count)
                                                                    .setStageFlags(vk::ShaderStageFlagBits::eFragment)
                                                                    .setPImmutableSamplers(nullptr)};

        auto const descriptor_layout = vk::DescriptorSetLayoutCreateInfo().setBindingCount(2).setPBindings(layout_bindings);

        auto result = device.createDescriptorSetLayout(&descriptor_layout, nullptr, &desc_layout);
        VERIFY(result == vk::Result::eSuccess);

        auto const pPipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo().setSetLayoutCount(1).setPSetLayouts(&desc_layout);

        result = device.createPipelineLayout(&pPipelineLayoutCreateInfo, nullptr, &pipeline_layout);
        VERIFY(result == vk::Result::eSuccess);

        return true;
    }    

    bool prepare_render_pass() {
        // The initial layout for the color and depth attachments will be LAYOUT_UNDEFINED
        // because at the start of the renderpass, we don't care about their contents.
        // At the start of the subpass, the color attachment's layout will be transitioned
        // to LAYOUT_COLOR_ATTACHMENT_OPTIMAL and the depth stencil attachment's layout
        // will be transitioned to LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL.  At the end of
        // the renderpass, the color attachment's layout will be transitioned to
        // LAYOUT_PRESENT_SRC_KHR to be ready to present.  This is all done as part of
        // the renderpass, no barriers are necessary.
        const vk::AttachmentDescription attachments[2] = {vk::AttachmentDescription()
                                                            .setFormat(format)
                                                            .setSamples(vk::SampleCountFlagBits::e1)
                                                            .setLoadOp(vk::AttachmentLoadOp::eClear)
                                                            .setStoreOp(vk::AttachmentStoreOp::eStore)
                                                            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                                            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                                            .setInitialLayout(vk::ImageLayout::eUndefined)
                                                            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR),
                                                        vk::AttachmentDescription()
                                                            .setFormat(depth.format)
                                                            .setSamples(vk::SampleCountFlagBits::e1)
                                                            .setLoadOp(vk::AttachmentLoadOp::eClear)
                                                            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                                                            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                                            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                                            .setInitialLayout(vk::ImageLayout::eUndefined)
                                                            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)};

        auto const color_reference = vk::AttachmentReference().setAttachment(0).setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        auto const depth_reference =
            vk::AttachmentReference().setAttachment(1).setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        auto const subpass = vk::SubpassDescription()
                                .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                .setInputAttachmentCount(0)
                                .setPInputAttachments(nullptr)
                                .setColorAttachmentCount(1)
                                .setPColorAttachments(&color_reference)
                                .setPResolveAttachments(nullptr)
                                .setPDepthStencilAttachment(&depth_reference)
                                .setPreserveAttachmentCount(0)
                                .setPPreserveAttachments(nullptr);

        vk::PipelineStageFlags stages = vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
        vk::SubpassDependency const dependencies[2] = {
            vk::SubpassDependency()  // Depth buffer is shared between swapchain images
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setDstSubpass(0)
                .setSrcStageMask(stages)
                .setDstStageMask(stages)
                .setSrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentWrite)
                .setDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite)
                .setDependencyFlags(vk::DependencyFlags()),
            vk::SubpassDependency()  // Image layout transition
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setDstSubpass(0)
                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlagBits())
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead)
                .setDependencyFlags(vk::DependencyFlags()),
        };

        auto const rp_info = vk::RenderPassCreateInfo()
                                .setAttachmentCount(2)
                                .setPAttachments(attachments)
                                .setSubpassCount(1)
                                .setPSubpasses(&subpass)
                                .setDependencyCount(2)
                                .setPDependencies(dependencies);

        auto result = device.createRenderPass(&rp_info, nullptr, &render_pass);
        VERIFY(result == vk::Result::eSuccess);

        return true;
    }    

    vk::ShaderModule prepare_shader_module(const uint32_t *code, size_t size) {
        const auto moduleCreateInfo = vk::ShaderModuleCreateInfo().setCodeSize(size).setPCode(code);

        vk::ShaderModule module;
        auto result = device.createShaderModule(&moduleCreateInfo, nullptr, &module);
        VERIFY(result == vk::Result::eSuccess);

        return module;
    }

    vk::ShaderModule prepare_vs() {
        const uint32_t vertShaderCode[] = {
    #include "cube.vert.inc"
        };

        vert_shader_module = prepare_shader_module(vertShaderCode, sizeof(vertShaderCode));

        return vert_shader_module;
    }

    vk::ShaderModule prepare_fs() {
        const uint32_t fragShaderCode[] = {
    #include "cube.frag.inc"
        };

        frag_shader_module = prepare_shader_module(fragShaderCode, sizeof(fragShaderCode));

        return frag_shader_module;
    }

    bool prepare_pipeline() {
        vk::PipelineCacheCreateInfo const pipelineCacheInfo;
        auto result = device.createPipelineCache(&pipelineCacheInfo, nullptr, &pipelineCache);
        VERIFY(result == vk::Result::eSuccess);

        vk::PipelineShaderStageCreateInfo const shaderStageInfo[2] = {
            vk::PipelineShaderStageCreateInfo().setStage(vk::ShaderStageFlagBits::eVertex).setModule(prepare_vs()).setPName("main"),
            vk::PipelineShaderStageCreateInfo().setStage(vk::ShaderStageFlagBits::eFragment).setModule(prepare_fs()).setPName("main")};

        vk::PipelineVertexInputStateCreateInfo const vertexInputInfo;

        auto const inputAssemblyInfo = vk::PipelineInputAssemblyStateCreateInfo().setTopology(vk::PrimitiveTopology::eTriangleList);

        // TODO: Where are pViewports and pScissors set?
        auto const viewportInfo = vk::PipelineViewportStateCreateInfo().setViewportCount(1).setScissorCount(1);

        auto const rasterizationInfo = vk::PipelineRasterizationStateCreateInfo()
                                        .setDepthClampEnable(VK_FALSE)
                                        .setRasterizerDiscardEnable(VK_FALSE)
                                        .setPolygonMode(vk::PolygonMode::eFill)
                                        .setCullMode(vk::CullModeFlagBits::eBack)
                                        .setFrontFace(vk::FrontFace::eCounterClockwise)
                                        .setDepthBiasEnable(VK_FALSE)
                                        .setLineWidth(1.0f);

        auto const multisampleInfo = vk::PipelineMultisampleStateCreateInfo();

        auto const stencilOp =
            vk::StencilOpState().setFailOp(vk::StencilOp::eKeep).setPassOp(vk::StencilOp::eKeep).setCompareOp(vk::CompareOp::eAlways);

        auto const depthStencilInfo = vk::PipelineDepthStencilStateCreateInfo()
                                        .setDepthTestEnable(VK_TRUE)
                                        .setDepthWriteEnable(VK_TRUE)
                                        .setDepthCompareOp(vk::CompareOp::eLessOrEqual)
                                        .setDepthBoundsTestEnable(VK_FALSE)
                                        .setStencilTestEnable(VK_FALSE)
                                        .setFront(stencilOp)
                                        .setBack(stencilOp);

        vk::PipelineColorBlendAttachmentState const colorBlendAttachments[1] = {
            vk::PipelineColorBlendAttachmentState().setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                                    vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)};

        auto const colorBlendInfo =
            vk::PipelineColorBlendStateCreateInfo().setAttachmentCount(1).setPAttachments(colorBlendAttachments);

        vk::DynamicState const dynamicStates[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        auto const dynamicStateInfo = vk::PipelineDynamicStateCreateInfo().setPDynamicStates(dynamicStates).setDynamicStateCount(2);

        auto const pipeline = vk::GraphicsPipelineCreateInfo()
                                .setStageCount(2)
                                .setPStages(shaderStageInfo)
                                .setPVertexInputState(&vertexInputInfo)
                                .setPInputAssemblyState(&inputAssemblyInfo)
                                .setPViewportState(&viewportInfo)
                                .setPRasterizationState(&rasterizationInfo)
                                .setPMultisampleState(&multisampleInfo)
                                .setPDepthStencilState(&depthStencilInfo)
                                .setPColorBlendState(&colorBlendInfo)
                                .setPDynamicState(&dynamicStateInfo)
                                .setLayout(pipeline_layout)
                                .setRenderPass(render_pass);

        result = device.createGraphicsPipelines(pipelineCache, 1, &pipeline, nullptr, &this->pipeline);
        VERIFY(result == vk::Result::eSuccess);

        device.destroyShaderModule(frag_shader_module, nullptr);
        device.destroyShaderModule(vert_shader_module, nullptr);

        return true;
    }    

    bool prepare_descriptor_pool() {
        vk::DescriptorPoolSize const poolSizes[2] = {
            vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(swapchainImageCount),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(swapchainImageCount * texture_count)};

        auto const descriptor_pool =
            vk::DescriptorPoolCreateInfo().setMaxSets(swapchainImageCount).setPoolSizeCount(2).setPPoolSizes(poolSizes);

        auto result = device.createDescriptorPool(&descriptor_pool, nullptr, &desc_pool);
        VERIFY(result == vk::Result::eSuccess);

        return true;
    }

    bool prepare_descriptor_set() {
        auto const alloc_info =
            vk::DescriptorSetAllocateInfo().setDescriptorPool(desc_pool).setDescriptorSetCount(1).setPSetLayouts(&desc_layout);

        auto buffer_info = vk::DescriptorBufferInfo().setOffset(0).setRange(sizeof(struct vktexcube_vs_uniform));

        vk::DescriptorImageInfo tex_descs[texture_count];
        for (uint32_t i = 0; i < texture_count; i++) {
            tex_descs[i].setSampler(textures[i].sampler);
            tex_descs[i].setImageView(textures[i].view);
            tex_descs[i].setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        }

        vk::WriteDescriptorSet writes[2];

        writes[0].setDescriptorCount(1);
        writes[0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
        writes[0].setPBufferInfo(&buffer_info);

        writes[1].setDstBinding(1);
        writes[1].setDescriptorCount(texture_count);
        writes[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
        writes[1].setPImageInfo(tex_descs);

        for (unsigned int i = 0; i < swapchainImageCount; i++) {
            auto result = device.allocateDescriptorSets(&alloc_info, &swapchain_image_resources[i].descriptor_set);
            VERIFY(result == vk::Result::eSuccess);

            buffer_info.setBuffer(swapchain_image_resources[i].uniform_buffer);
            writes[0].setDstSet(swapchain_image_resources[i].descriptor_set);
            writes[1].setDstSet(swapchain_image_resources[i].descriptor_set);
            device.updateDescriptorSets(2, writes, 0, nullptr);
        }

        return true;
    }

    bool prepare_framebuffers() {
        vk::ImageView attachments[2];
        attachments[1] = depth.view;

        auto const fb_info = vk::FramebufferCreateInfo()
                                .setRenderPass(render_pass)
                                .setAttachmentCount(2)
                                .setPAttachments(attachments)
                                .setWidth((uint32_t)width)
                                .setHeight((uint32_t)height)
                                .setLayers(1);

        for (uint32_t i = 0; i < swapchainImageCount; i++) {
            attachments[0] = swapchain_image_resources[i].view;
            auto const result = device.createFramebuffer(&fb_info, nullptr, &swapchain_image_resources[i].framebuffer);
            VERIFY(result == vk::Result::eSuccess);
        }

        return true;
    }

    bool allocate_command_buffers() {
        for (uint32_t i = 0; i < swapchainImageCount; ++i) {
            auto result = device.allocateCommandBuffers(&cmd_alloc_info, &swapchain_image_resources[i].cmd);
            VERIFY(result == vk::Result::eSuccess);
        }

        if (separate_present_queue) {
            auto const present_cmd_pool_info = vk::CommandPoolCreateInfo().setQueueFamilyIndex(present_queue_family_index);

            auto result = device.createCommandPool(&present_cmd_pool_info, nullptr, &present_cmd_pool);
            VERIFY(result == vk::Result::eSuccess);

            auto const present_cmd = vk::CommandBufferAllocateInfo()
                                        .setCommandPool(present_cmd_pool)
                                        .setLevel(vk::CommandBufferLevel::ePrimary)
                                        .setCommandBufferCount(1);

            for (uint32_t i = 0; i < swapchainImageCount; i++) {
                result = device.allocateCommandBuffers(&present_cmd, &swapchain_image_resources[i].graphics_to_present_cmd);
                VERIFY(result == vk::Result::eSuccess);

                build_image_ownership_cmd(i);
            }
        }     

        return true;   
    }

    bool draw_build_cmd() {
        for (uint32_t i = 0; i < swapchainImageCount; ++i) {
            current_buffer = i;
            draw_build_cmd(swapchain_image_resources[i].cmd);
        }

        return true;
    }

    bool flush_init_cmd() {
        // TODO: hmm.
        // This function could get called twice if the texture uses a staging
        // buffer
        // In that case the second call should be ignored
        if (!cmd) {
            return false;
        }

        auto result = cmd.end();
        VERIFY(result == vk::Result::eSuccess);

        auto const fenceInfo = vk::FenceCreateInfo();
        vk::Fence fence;
        result = device.createFence(&fenceInfo, nullptr, &fence);
        VERIFY(result == vk::Result::eSuccess);

        vk::CommandBuffer const commandBuffers[] = {cmd};
        auto const submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(commandBuffers);

        result = graphics_queue.submit(1, &submitInfo, fence);
        VERIFY(result == vk::Result::eSuccess);

        result = device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);
        VERIFY(result == vk::Result::eSuccess);

        device.freeCommandBuffers(cmd_pool, 1, commandBuffers);
        device.destroyFence(fence, nullptr);

        cmd = vk::CommandBuffer();        

        return true;
    }

    void destroy_texture(texture_object *tex_objs) {
        // clean up staging resources
        device.freeMemory(tex_objs->mem, nullptr);
        if (tex_objs->image) device.destroyImage(tex_objs->image, nullptr);
        if (tex_objs->buffer) device.destroyBuffer(tex_objs->buffer, nullptr);
    }

    bool destroy_textures() {
        if (staging_texture.buffer) {
            destroy_texture(&staging_texture);
        }

        return true;
    }

    void build_image_ownership_cmd(uint32_t i) {
        auto const cmd_buf_info = vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        auto result = swapchain_image_resources[i].graphics_to_present_cmd.begin(&cmd_buf_info);
        VERIFY(result == vk::Result::eSuccess);

        auto const image_ownership_barrier =
            vk::ImageMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlags())
                .setDstAccessMask(vk::AccessFlags())
                .setOldLayout(vk::ImageLayout::ePresentSrcKHR)
                .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
                .setSrcQueueFamilyIndex(graphics_queue_family_index)
                .setDstQueueFamilyIndex(present_queue_family_index)
                .setImage(swapchain_image_resources[i].image)
                .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

        swapchain_image_resources[i].graphics_to_present_cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlagBits(), 0, nullptr, 0,
            nullptr, 1, &image_ownership_barrier);

        result = swapchain_image_resources[i].graphics_to_present_cmd.end();
        VERIFY(result == vk::Result::eSuccess);
    }    

    void draw() {
        // Ensure no more than FRAME_LAG renderings are outstanding
        device.waitForFences(1, &fences[frame_index], VK_TRUE, UINT64_MAX);
        device.resetFences(1, &fences[frame_index]);

        vk::Result result;
        do {
            result =
                device.acquireNextImageKHR(swapchain, UINT64_MAX, image_acquired_semaphores[frame_index], vk::Fence(), &current_buffer);
            if (result == vk::Result::eErrorOutOfDateKHR) {
                // this->swapchain is out of date (e.g. the window was resized) and
                // must be recreated:
                resize();
            } else if (result == vk::Result::eSuboptimalKHR) {
                // swapchain is not as optimal as it could be, but the platform's
                // presentation engine will still present the image correctly.
                break;
            } else if (result == vk::Result::eErrorSurfaceLostKHR) {
                inst.destroySurfaceKHR(surface, nullptr);
                create_surface();
                resize();
            } else {
                VERIFY(result == vk::Result::eSuccess);
            }
        } while (result != vk::Result::eSuccess);

        update_data_buffer();

        // Wait for the image acquired semaphore to be signaled to ensure
        // that the image won't be rendered to until the presentation
        // engine has fully released ownership to the application, and it is
        // okay to render to the image.
        vk::PipelineStageFlags const pipe_stage_flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        auto const submit_info = vk::SubmitInfo()
                                    .setPWaitDstStageMask(&pipe_stage_flags)
                                    .setWaitSemaphoreCount(1)
                                    .setPWaitSemaphores(&image_acquired_semaphores[frame_index])
                                    .setCommandBufferCount(1)
                                    .setPCommandBuffers(&swapchain_image_resources[current_buffer].cmd)
                                    .setSignalSemaphoreCount(1)
                                    .setPSignalSemaphores(&draw_complete_semaphores[frame_index]);

        result = graphics_queue.submit(1, &submit_info, fences[frame_index]);
        VERIFY(result == vk::Result::eSuccess);

        if (separate_present_queue) {
            // If we are using separate queues, change image ownership to the
            // present queue before presenting, waiting for the draw complete
            // semaphore and signalling the ownership released semaphore when
            // finished
            auto const present_submit_info = vk::SubmitInfo()
                                                .setPWaitDstStageMask(&pipe_stage_flags)
                                                .setWaitSemaphoreCount(1)
                                                .setPWaitSemaphores(&draw_complete_semaphores[frame_index])
                                                .setCommandBufferCount(1)
                                                .setPCommandBuffers(&swapchain_image_resources[current_buffer].graphics_to_present_cmd)
                                                .setSignalSemaphoreCount(1)
                                                .setPSignalSemaphores(&image_ownership_semaphores[frame_index]);

            result = present_queue.submit(1, &present_submit_info, vk::Fence());
            VERIFY(result == vk::Result::eSuccess);
        }

        // If we are using separate queues we have to wait for image ownership,
        // otherwise wait for draw complete
        auto const presentInfo = vk::PresentInfoKHR()
                                    .setWaitSemaphoreCount(1)
                                    .setPWaitSemaphores(separate_present_queue ? &image_ownership_semaphores[frame_index]
                                                                                : &draw_complete_semaphores[frame_index])
                                    .setSwapchainCount(1)
                                    .setPSwapchains(&swapchain)
                                    .setPImageIndices(&current_buffer);

        result = present_queue.presentKHR(&presentInfo);
        frame_index += 1;
        frame_index %= FRAME_LAG;
        if (result == vk::Result::eErrorOutOfDateKHR) {
            // swapchain is out of date (e.g. the window was resized) and
            // must be recreated:
            resize();
        } else if (result == vk::Result::eSuboptimalKHR) {
            // swapchain is not as optimal as it could be, but the platform's
            // presentation engine will still present the image correctly.
        } else if (result == vk::Result::eErrorSurfaceLostKHR) {
            inst.destroySurfaceKHR(surface, nullptr);
            create_surface();
            resize();
        } else {
            VERIFY(result == vk::Result::eSuccess);
        }
    }

    void draw_build_cmd(vk::CommandBuffer commandBuffer) {
        auto const commandInfo = vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

        vk::ClearValue const clearValues[2] = {vk::ClearColorValue(std::array<float, 4>({{0.2f, 0.2f, 0.2f, 0.2f}})),
                                            vk::ClearDepthStencilValue(1.0f, 0u)};

        auto const passInfo = vk::RenderPassBeginInfo()
                                .setRenderPass(render_pass)
                                .setFramebuffer(swapchain_image_resources[current_buffer].framebuffer)
                                .setRenderArea(vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D((uint32_t)width, (uint32_t)height)))
                                .setClearValueCount(2)
                                .setPClearValues(clearValues);

        auto result = commandBuffer.begin(&commandInfo);
        VERIFY(result == vk::Result::eSuccess);

        commandBuffer.beginRenderPass(&passInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1,
                                        &swapchain_image_resources[current_buffer].descriptor_set, 0, nullptr);
        float viewport_dimension;
        float viewport_x = 0.0f;
        float viewport_y = 0.0f;
        if (width < height) {
            viewport_dimension = (float)width;
            viewport_y = (height - width) / 2.0f;
        } else {
            viewport_dimension = (float)height;
            viewport_x = (width - height) / 2.0f;
        }
        auto const viewport = vk::Viewport()
                                .setX(viewport_x)
                                .setY(viewport_y)
                                .setWidth((float)viewport_dimension)
                                .setHeight((float)viewport_dimension)
                                .setMinDepth((float)0.0f)
                                .setMaxDepth((float)1.0f);
        commandBuffer.setViewport(0, 1, &viewport);

        vk::Rect2D const scissor(vk::Offset2D(0, 0), vk::Extent2D(width, height));
        commandBuffer.setScissor(0, 1, &scissor);
        commandBuffer.draw(12 * 3, 1, 0, 0);
        // Note that ending the renderpass changes the image's layout from
        // COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
        commandBuffer.endRenderPass();

        if (separate_present_queue) {
            // We have to transfer ownership from the graphics queue family to
            // the
            // present queue family to be able to present.  Note that we don't
            // have
            // to transfer from present queue family back to graphics queue
            // family at
            // the start of the next frame because we don't care about the
            // image's
            // contents at that point.
            auto const image_ownership_barrier =
                vk::ImageMemoryBarrier()
                    .setSrcAccessMask(vk::AccessFlags())
                    .setDstAccessMask(vk::AccessFlags())
                    .setOldLayout(vk::ImageLayout::ePresentSrcKHR)
                    .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
                    .setSrcQueueFamilyIndex(graphics_queue_family_index)
                    .setDstQueueFamilyIndex(present_queue_family_index)
                    .setImage(swapchain_image_resources[current_buffer].image)
                    .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eBottomOfPipe,
                                        vk::DependencyFlagBits(), 0, nullptr, 0, nullptr, 1, &image_ownership_barrier);
        }

        result = commandBuffer.end();
        VERIFY(result == vk::Result::eSuccess);
    }

    bool memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex) {
        // Search memtypes to find first index with those properties
        for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; i++) {
            if ((typeBits & 1) == 1) {
                // Type is available, does it match user properties?
                if ((memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                    *typeIndex = i;
                    return true;
                }
            }
            typeBits >>= 1;
        }

        // No memory types matched, return failure
        return false;
    }

    void error(std::string_view msg, std::string_view group) {
        std::cerr << group << ": " << msg << std::endl;
        exit(1);
    }
};

VulkanApi vulkanApi;
extern HINSTANCE instance;

extern "C"
{

void create_vulkan(intptr_t hwnd) {
    vulkanApi.initialize();
    vulkanApi.create_surface_win32(instance, (HWND) hwnd);
    vulkanApi.initialize_swapchain();
    vulkanApi.prepare();
}

void run_vulkan() {
    vulkanApi.run();
}

void cleanup_vulkan() {
    vulkanApi.cleanup();
}

}