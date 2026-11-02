#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/rendering/Events.hh>
#include <ignition/common/Event.hh>
#include <ignition/msgs/image.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/rendering/ogre2/Ogre2Material.hh>
#include <ignition/rendering/RenderEngine.hh>
#include <ignition/rendering/RenderEngineManager.hh>
#include <ignition/rendering/Scene.hh>
#include <ignition/rendering/Visual.hh>
#include <ignition/transport/Node.hh>

#include <OgreHlmsPbsDatablock.h>
#include <OgreStagingTexture.h>
#include <OgreTextureGpu.h>
#include <OgreTextureGpuManager.h>

namespace
{
class TextureProbeSystem final : public ignition::gazebo::System,
                                 public ignition::gazebo::ISystemConfigure
{
  public: void Configure(
      const ignition::gazebo::Entity &,
      const std::shared_ptr<const sdf::Element> &_sdf,
      ignition::gazebo::EntityComponentManager &,
      ignition::gazebo::EventManager &_eventManager) override
  {
    this->topic = _sdf->Get<std::string>("topic", "/texture_probe/update").first;
    if (!this->transportNode.Subscribe(
        this->topic, &TextureProbeSystem::OnImage, this))
    {
      std::cerr << "texture probe: failed to subscribe to " << this->topic
                << std::endl;
    }
    this->preRenderConnection = _eventManager.Connect<
        ignition::gazebo::events::PreRender>(
        std::bind(&TextureProbeSystem::OnPreRender, this));
  }

  private: void OnImage(const ignition::msgs::Image &_image)
  {
    if (_image.width() != 256 || _image.height() != 256 ||
        _image.pixel_format_type() != ignition::msgs::RGB_INT8 ||
        _image.step() < _image.width() * 3 ||
        _image.data().size() < _image.step() * _image.height())
    {
      std::cerr << "texture probe: expected 256x256 RGB_INT8 image"
                << std::endl;
      return;
    }

    std::lock_guard<std::mutex> lock(this->imageMutex);
    this->latestImage.assign(_image.data().begin(), _image.data().end());
    this->latestStep = _image.step();
    ++this->imageSequence;
  }

  private: void OnPreRender()
  {
    std::vector<uint8_t> rgb;
    uint32_t step;
    uint64_t sequence;
    {
      std::lock_guard<std::mutex> lock(this->imageMutex);
      if (this->imageSequence == 0 ||
          this->imageSequence == this->uploadedSequence)
      {
        return;
      }
      rgb = this->latestImage;
      step = this->latestStep;
      sequence = this->imageSequence;
    }

    auto *manager = ignition::rendering::RenderEngineManager::Instance();
    auto *engine = manager->Engine("ogre2");
    if (!engine)
      return;

    unsigned int uploadedSceneCount = 0;
    uint8_t uploadedMipCount = 0;
    for (unsigned int index = 0; index < engine->SceneCount(); ++index)
    {
      auto scene = engine->SceneByIndex(index);
      ignition::rendering::VisualPtr visual;
      for (unsigned int nodeIndex = 0;
           nodeIndex < scene->NodeCount(); ++nodeIndex)
      {
        auto node = scene->NodeByIndex(nodeIndex);
        if (node && node->Name() == "paper::link::surface")
        {
          visual = std::dynamic_pointer_cast<
              ignition::rendering::Visual>(node);
          break;
        }
      }
      if (!visual)
        continue;

      auto material = visual->Material();
      if (!material && visual->GeometryCount() > 0)
        material = visual->GeometryByIndex(0)->Material();
      if (!material)
        continue;

      auto *ogreMaterial = dynamic_cast<ignition::rendering::Ogre2Material *>(
          material.get());
      auto *datablock = ogreMaterial ? ogreMaterial->Datablock() : nullptr;
      auto *texture = datablock ? datablock->getTexture(Ogre::PBSM_DIFFUSE) : nullptr;
      if (!texture || texture->getWidth() != 256 || texture->getHeight() != 256)
      {
        std::cerr << "texture probe: paper albedo texture unavailable"
                  << std::endl;
        continue;
      }

      auto format = texture->getPixelFormat();
      if (format != Ogre::PFG_RGBA8_UNORM &&
          format != Ogre::PFG_RGBA8_UNORM_SRGB)
      {
        std::cerr << "texture probe: unsupported GPU pixel format "
                  << static_cast<int>(format) << std::endl;
        continue;
      }

      std::vector<uint8_t> rgba(256 * 256 * 4);
      for (uint32_t y = 0; y < 256; ++y)
      {
        const auto *source = rgb.data() + y * step;
        auto *destination = rgba.data() + y * 256 * 4;
        for (uint32_t x = 0; x < 256; ++x)
        {
          destination[x * 4 + 0] = source[x * 3 + 0];
          destination[x * 4 + 1] = source[x * 3 + 1];
          destination[x * 4 + 2] = source[x * 3 + 2];
          destination[x * 4 + 3] = 255;
        }
      }

      auto *textureManager = texture->getTextureManager();
      if (!this->stagingTexture)
      {
        this->stagingTexture = textureManager->getStagingTexture(
            256, 256, 1, 1, format);
      }
      auto *staging = this->stagingTexture;
      if (!staging)
      {
        std::cerr << "texture probe: staging texture unavailable" << std::endl;
        continue;
      }
      uint32_t mipWidth = 256;
      uint32_t mipHeight = 256;
      const auto mipCount = texture->getNumMipmaps();
      bool uploadFailed = false;
      for (uint8_t mip = 0; mip < mipCount; ++mip)
      {
        staging->startMapRegion();
        auto box = staging->mapRegion(mipWidth, mipHeight, 1, 1, format);
        if (!box.data)
        {
          staging->stopMapRegion();
          std::cerr << "texture probe: staging map failed at mip "
                    << static_cast<int>(mip) << std::endl;
          uploadFailed = true;
          break;
        }
        for (uint32_t y = 0; y < mipHeight; ++y)
        {
          auto *destination = static_cast<uint8_t *>(box.data) + y * box.bytesPerRow;
          std::memcpy(destination, rgba.data() + y * mipWidth * 4, mipWidth * 4);
        }
        staging->stopMapRegion();
        Ogre::TextureBox destinationBox(
            mipWidth, mipHeight, 1, 1, 4, mipWidth * 4,
            mipWidth * mipHeight * 4);
        destinationBox.sliceStart = texture->getInternalSliceStart();
        staging->upload(box, texture, mip, nullptr, &destinationBox);

        if (mip + 1 < mipCount)
        {
          const uint32_t nextWidth = std::max(1u, mipWidth / 2);
          const uint32_t nextHeight = std::max(1u, mipHeight / 2);
          std::vector<uint8_t> nextRgba(nextWidth * nextHeight * 4);
          for (uint32_t y = 0; y < nextHeight; ++y)
          {
            for (uint32_t x = 0; x < nextWidth; ++x)
            {
              for (uint32_t channel = 0; channel < 4; ++channel)
              {
                uint32_t sum = 0;
                for (uint32_t dy = 0; dy < 2; ++dy)
                {
                  for (uint32_t dx = 0; dx < 2; ++dx)
                  {
                    const auto sourceIndex =
                        ((y * 2 + dy) * mipWidth + x * 2 + dx) * 4 + channel;
                    sum += rgba[sourceIndex];
                  }
                }
                nextRgba[(y * nextWidth + x) * 4 + channel] =
                    static_cast<uint8_t>(sum / 4);
              }
            }
          }
          rgba = std::move(nextRgba);
          mipWidth = nextWidth;
          mipHeight = nextHeight;
        }
      }
      if (uploadFailed)
        continue;
      ++uploadedSceneCount;
      uploadedMipCount = mipCount;
    }

    if (uploadedSceneCount == 0)
      return;

    this->uploadedSequence = sequence;
    if (sequence == 1)
      std::cerr << "texture probe: uploaded image sequence " << sequence
                << " directly to GPU in " << uploadedSceneCount
                << " scene(s), including "
                << static_cast<int>(uploadedMipCount)
                << " mip levels" << std::endl;
  }

  private: ignition::transport::Node transportNode;
  private: std::string topic;
  private: ignition::common::ConnectionPtr preRenderConnection;
  private: std::mutex imageMutex;
  private: std::vector<uint8_t> latestImage;
  private: uint32_t latestStep{0};
  private: uint64_t imageSequence{0};
  private: uint64_t uploadedSequence{0};
  private: Ogre::StagingTexture *stagingTexture{nullptr};
};
}

IGNITION_ADD_PLUGIN(TextureProbeSystem,
  ignition::gazebo::System,
  ignition::gazebo::ISystemConfigure)
IGNITION_ADD_PLUGIN_ALIAS(TextureProbeSystem,
    "texture_probe::TextureProbeSystem")