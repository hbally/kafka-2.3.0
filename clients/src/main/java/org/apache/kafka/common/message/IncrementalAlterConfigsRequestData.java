/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// THIS CODE IS AUTOMATICALLY GENERATED.  DO NOT EDIT.

package org.apache.kafka.common.message;

import java.util.Iterator;
import java.util.List;
import org.apache.kafka.common.protocol.ApiMessage;
import org.apache.kafka.common.protocol.Message;
import org.apache.kafka.common.protocol.MessageUtil;
import org.apache.kafka.common.protocol.Readable;
import org.apache.kafka.common.protocol.Writable;
import org.apache.kafka.common.protocol.types.ArrayOf;
import org.apache.kafka.common.protocol.types.Field;
import org.apache.kafka.common.protocol.types.Schema;
import org.apache.kafka.common.protocol.types.Struct;
import org.apache.kafka.common.protocol.types.Type;
import org.apache.kafka.common.utils.ImplicitLinkedHashMultiCollection;


public class IncrementalAlterConfigsRequestData implements ApiMessage {
    private AlterConfigsResourceCollection resources;
    private boolean validateOnly;
    
    public static final Schema SCHEMA_0 =
        new Schema(
            new Field("resources", new ArrayOf(AlterConfigsResource.SCHEMA_0), "The incremental updates for each resource."),
            new Field("validate_only", Type.BOOLEAN, "True if we should validate the request, but not change the configurations.")
        );
    
    public static final Schema[] SCHEMAS = new Schema[] {
        SCHEMA_0
    };
    
    public IncrementalAlterConfigsRequestData(Readable readable, short version) {
        this.resources = new AlterConfigsResourceCollection(0);
        read(readable, version);
    }
    
    public IncrementalAlterConfigsRequestData(Struct struct, short version) {
        this.resources = new AlterConfigsResourceCollection(0);
        fromStruct(struct, version);
    }
    
    public IncrementalAlterConfigsRequestData() {
        this.resources = new AlterConfigsResourceCollection(0);
        this.validateOnly = false;
    }
    
    @Override
    public short apiKey() {
        return 44;
    }
    
    @Override
    public short lowestSupportedVersion() {
        return 0;
    }
    
    @Override
    public short highestSupportedVersion() {
        return 0;
    }
    
    @Override
    public void read(Readable readable, short version) {
        {
            int arrayLength = readable.readInt();
            if (arrayLength < 0) {
                this.resources = null;
            } else {
                this.resources.clear(arrayLength);
                for (int i = 0; i < arrayLength; i++) {
                    this.resources.add(new AlterConfigsResource(readable, version));
                }
            }
        }
        this.validateOnly = readable.readByte() != 0;
    }
    
    @Override
    public void write(Writable writable, short version) {
        writable.writeInt(resources.size());
        for (AlterConfigsResource element : resources) {
            element.write(writable, version);
        }
        writable.writeByte(validateOnly ? (byte) 1 : (byte) 0);
    }
    
    @Override
    public void fromStruct(Struct struct, short version) {
        {
            Object[] nestedObjects = struct.getArray("resources");
            this.resources = new AlterConfigsResourceCollection(nestedObjects.length);
            for (Object nestedObject : nestedObjects) {
                this.resources.add(new AlterConfigsResource((Struct) nestedObject, version));
            }
        }
        this.validateOnly = struct.getBoolean("validate_only");
    }
    
    @Override
    public Struct toStruct(short version) {
        Struct struct = new Struct(SCHEMAS[version]);
        {
            Struct[] nestedObjects = new Struct[resources.size()];
            int i = 0;
            for (AlterConfigsResource element : this.resources) {
                nestedObjects[i++] = element.toStruct(version);
            }
            struct.set("resources", (Object[]) nestedObjects);
        }
        struct.set("validate_only", this.validateOnly);
        return struct;
    }
    
    @Override
    public int size(short version) {
        int size = 0;
        size += 4;
        for (AlterConfigsResource element : resources) {
            size += element.size(version);
        }
        size += 1;
        return size;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof IncrementalAlterConfigsRequestData)) return false;
        IncrementalAlterConfigsRequestData other = (IncrementalAlterConfigsRequestData) obj;
        if (this.resources == null) {
            if (other.resources != null) return false;
        } else {
            if (!this.resources.equals(other.resources)) return false;
        }
        if (validateOnly != other.validateOnly) return false;
        return true;
    }
    
    @Override
    public int hashCode() {
        int hashCode = 0;
        hashCode = 31 * hashCode + (resources == null ? 0 : resources.hashCode());
        hashCode = 31 * hashCode + (validateOnly ? 1231 : 1237);
        return hashCode;
    }
    
    @Override
    public String toString() {
        return "IncrementalAlterConfigsRequestData("
            + "resources=" + MessageUtil.deepToString(resources.iterator())
            + ", validateOnly=" + (validateOnly ? "true" : "false")
            + ")";
    }
    
    public AlterConfigsResourceCollection resources() {
        return this.resources;
    }
    
    public boolean validateOnly() {
        return this.validateOnly;
    }
    
    public IncrementalAlterConfigsRequestData setResources(AlterConfigsResourceCollection v) {
        this.resources = v;
        return this;
    }
    
    public IncrementalAlterConfigsRequestData setValidateOnly(boolean v) {
        this.validateOnly = v;
        return this;
    }
    
    static public class AlterConfigsResource implements Message, ImplicitLinkedHashMultiCollection.Element {
        private byte resourceType;
        private String resourceName;
        private AlterableConfigCollection configs;
        private int next;
        private int prev;
        
        public static final Schema SCHEMA_0 =
            new Schema(
                new Field("resource_type", Type.INT8, "The resource type."),
                new Field("resource_name", Type.STRING, "The resource name."),
                new Field("configs", new ArrayOf(AlterableConfig.SCHEMA_0), "The configurations.")
            );
        
        public static final Schema[] SCHEMAS = new Schema[] {
            SCHEMA_0
        };
        
        public AlterConfigsResource(Readable readable, short version) {
            this.configs = new AlterableConfigCollection(0);
            read(readable, version);
        }
        
        public AlterConfigsResource(Struct struct, short version) {
            this.configs = new AlterableConfigCollection(0);
            fromStruct(struct, version);
        }
        
        public AlterConfigsResource() {
            this.resourceType = (byte) 0;
            this.resourceName = "";
            this.configs = new AlterableConfigCollection(0);
        }
        
        
        @Override
        public short lowestSupportedVersion() {
            return 0;
        }
        
        @Override
        public short highestSupportedVersion() {
            return 0;
        }
        
        @Override
        public void read(Readable readable, short version) {
            this.resourceType = readable.readByte();
            this.resourceName = readable.readNullableString();
            {
                int arrayLength = readable.readInt();
                if (arrayLength < 0) {
                    this.configs = null;
                } else {
                    this.configs.clear(arrayLength);
                    for (int i = 0; i < arrayLength; i++) {
                        this.configs.add(new AlterableConfig(readable, version));
                    }
                }
            }
        }
        
        @Override
        public void write(Writable writable, short version) {
            writable.writeByte(resourceType);
            writable.writeString(resourceName);
            writable.writeInt(configs.size());
            for (AlterableConfig element : configs) {
                element.write(writable, version);
            }
        }
        
        @Override
        public void fromStruct(Struct struct, short version) {
            this.resourceType = struct.getByte("resource_type");
            this.resourceName = struct.getString("resource_name");
            {
                Object[] nestedObjects = struct.getArray("configs");
                this.configs = new AlterableConfigCollection(nestedObjects.length);
                for (Object nestedObject : nestedObjects) {
                    this.configs.add(new AlterableConfig((Struct) nestedObject, version));
                }
            }
        }
        
        @Override
        public Struct toStruct(short version) {
            Struct struct = new Struct(SCHEMAS[version]);
            struct.set("resource_type", this.resourceType);
            struct.set("resource_name", this.resourceName);
            {
                Struct[] nestedObjects = new Struct[configs.size()];
                int i = 0;
                for (AlterableConfig element : this.configs) {
                    nestedObjects[i++] = element.toStruct(version);
                }
                struct.set("configs", (Object[]) nestedObjects);
            }
            return struct;
        }
        
        @Override
        public int size(short version) {
            int size = 0;
            size += 1;
            size += 2;
            size += MessageUtil.serializedUtf8Length(resourceName);
            size += 4;
            for (AlterableConfig element : configs) {
                size += element.size(version);
            }
            return size;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof AlterConfigsResource)) return false;
            AlterConfigsResource other = (AlterConfigsResource) obj;
            if (resourceType != other.resourceType) return false;
            if (this.resourceName == null) {
                if (other.resourceName != null) return false;
            } else {
                if (!this.resourceName.equals(other.resourceName)) return false;
            }
            return true;
        }
        
        @Override
        public int hashCode() {
            int hashCode = 0;
            hashCode = 31 * hashCode + resourceType;
            hashCode = 31 * hashCode + (resourceName == null ? 0 : resourceName.hashCode());
            return hashCode;
        }
        
        @Override
        public String toString() {
            return "AlterConfigsResource("
                + "resourceType=" + resourceType
                + ", resourceName='" + resourceName + "'"
                + ", configs=" + MessageUtil.deepToString(configs.iterator())
                + ")";
        }
        
        public byte resourceType() {
            return this.resourceType;
        }
        
        public String resourceName() {
            return this.resourceName;
        }
        
        public AlterableConfigCollection configs() {
            return this.configs;
        }
        
        @Override
        public int next() {
            return this.next;
        }
        
        @Override
        public int prev() {
            return this.prev;
        }
        
        public AlterConfigsResource setResourceType(byte v) {
            this.resourceType = v;
            return this;
        }
        
        public AlterConfigsResource setResourceName(String v) {
            this.resourceName = v;
            return this;
        }
        
        public AlterConfigsResource setConfigs(AlterableConfigCollection v) {
            this.configs = v;
            return this;
        }
        
        @Override
        public void setNext(int v) {
            this.next = v;
        }
        
        @Override
        public void setPrev(int v) {
            this.prev = v;
        }
    }
    
    static public class AlterableConfig implements Message, ImplicitLinkedHashMultiCollection.Element {
        private String name;
        private byte configOperation;
        private String value;
        private int next;
        private int prev;
        
        public static final Schema SCHEMA_0 =
            new Schema(
                new Field("name", Type.STRING, "The configuration key name."),
                new Field("config_operation", Type.INT8, "The type (Set, Delete, Append, Subtract) of operation."),
                new Field("value", Type.NULLABLE_STRING, "The value to set for the configuration key.")
            );
        
        public static final Schema[] SCHEMAS = new Schema[] {
            SCHEMA_0
        };
        
        public AlterableConfig(Readable readable, short version) {
            read(readable, version);
        }
        
        public AlterableConfig(Struct struct, short version) {
            fromStruct(struct, version);
        }
        
        public AlterableConfig() {
            this.name = "";
            this.configOperation = (byte) 0;
            this.value = "";
        }
        
        
        @Override
        public short lowestSupportedVersion() {
            return 0;
        }
        
        @Override
        public short highestSupportedVersion() {
            return 0;
        }
        
        @Override
        public void read(Readable readable, short version) {
            this.name = readable.readNullableString();
            this.configOperation = readable.readByte();
            this.value = readable.readNullableString();
        }
        
        @Override
        public void write(Writable writable, short version) {
            writable.writeString(name);
            writable.writeByte(configOperation);
            writable.writeNullableString(value);
        }
        
        @Override
        public void fromStruct(Struct struct, short version) {
            this.name = struct.getString("name");
            this.configOperation = struct.getByte("config_operation");
            this.value = struct.getString("value");
        }
        
        @Override
        public Struct toStruct(short version) {
            Struct struct = new Struct(SCHEMAS[version]);
            struct.set("name", this.name);
            struct.set("config_operation", this.configOperation);
            struct.set("value", this.value);
            return struct;
        }
        
        @Override
        public int size(short version) {
            int size = 0;
            size += 2;
            size += MessageUtil.serializedUtf8Length(name);
            size += 1;
            size += 2;
            if (value != null) {
                size += MessageUtil.serializedUtf8Length(value);
            }
            return size;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof AlterableConfig)) return false;
            AlterableConfig other = (AlterableConfig) obj;
            if (this.name == null) {
                if (other.name != null) return false;
            } else {
                if (!this.name.equals(other.name)) return false;
            }
            if (configOperation != other.configOperation) return false;
            return true;
        }
        
        @Override
        public int hashCode() {
            int hashCode = 0;
            hashCode = 31 * hashCode + (name == null ? 0 : name.hashCode());
            hashCode = 31 * hashCode + configOperation;
            return hashCode;
        }
        
        @Override
        public String toString() {
            return "AlterableConfig("
                + "name='" + name + "'"
                + ", configOperation=" + configOperation
                + ", value='" + value + "'"
                + ")";
        }
        
        public String name() {
            return this.name;
        }
        
        public byte configOperation() {
            return this.configOperation;
        }
        
        public String value() {
            return this.value;
        }
        
        @Override
        public int next() {
            return this.next;
        }
        
        @Override
        public int prev() {
            return this.prev;
        }
        
        public AlterableConfig setName(String v) {
            this.name = v;
            return this;
        }
        
        public AlterableConfig setConfigOperation(byte v) {
            this.configOperation = v;
            return this;
        }
        
        public AlterableConfig setValue(String v) {
            this.value = v;
            return this;
        }
        
        @Override
        public void setNext(int v) {
            this.next = v;
        }
        
        @Override
        public void setPrev(int v) {
            this.prev = v;
        }
    }
    
    public static class AlterableConfigCollection extends ImplicitLinkedHashMultiCollection<AlterableConfig> {
        public AlterableConfigCollection() {
            super();
        }
        
        public AlterableConfigCollection(int expectedNumElements) {
            super(expectedNumElements);
        }
        
        public AlterableConfigCollection(Iterator<AlterableConfig> iterator) {
            super(iterator);
        }
        
        public AlterableConfig find(String name, byte configOperation) {
            AlterableConfig key = new AlterableConfig();
            key.setName(name);
            key.setConfigOperation(configOperation);
            return find(key);
        }
        
        public List<AlterableConfig> findAll(String name, byte configOperation) {
            AlterableConfig key = new AlterableConfig();
            key.setName(name);
            key.setConfigOperation(configOperation);
            return findAll(key);
        }
        
    }
    
    public static class AlterConfigsResourceCollection extends ImplicitLinkedHashMultiCollection<AlterConfigsResource> {
        public AlterConfigsResourceCollection() {
            super();
        }
        
        public AlterConfigsResourceCollection(int expectedNumElements) {
            super(expectedNumElements);
        }
        
        public AlterConfigsResourceCollection(Iterator<AlterConfigsResource> iterator) {
            super(iterator);
        }
        
        public AlterConfigsResource find(byte resourceType, String resourceName) {
            AlterConfigsResource key = new AlterConfigsResource();
            key.setResourceType(resourceType);
            key.setResourceName(resourceName);
            return find(key);
        }
        
        public List<AlterConfigsResource> findAll(byte resourceType, String resourceName) {
            AlterConfigsResource key = new AlterConfigsResource();
            key.setResourceType(resourceType);
            key.setResourceName(resourceName);
            return findAll(key);
        }
        
    }
}
